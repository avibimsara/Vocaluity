# =============================================================================
# One-time GCP bootstrap for Vocaluity
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - A GCP project already created
#
# Usage:
#   .\infra\bootstrap.ps1 -ProjectId <PROJECT_ID> -GitHubRepo <GITHUB_REPO>
#
# Example:
#   .\infra\bootstrap.ps1 -ProjectId my-gcp-project -GitHubRepo bimsarapathiraja/Vocaluity
#
# After running, add these 3 secrets to your GitHub repo:
#   GCP_PROJECT_ID, GCP_WIF_PROVIDER, GCP_SERVICE_ACCOUNT
# =============================================================================

param(
    [Parameter(Mandatory)][string]$ProjectId,
    [Parameter(Mandatory)][string]$GitHubRepo
)

$ErrorActionPreference = "Continue"
$Region = "us-central1"
$AppName = "vocaluity"

# Helper: run a command, ignore ALREADY_EXISTS, fail on other errors
function Invoke-Gcloud {
    param([string]$Step)
    # Caller pipes the command output into this; we just check $LASTEXITCODE after
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    WARNING in ${Step}: non-zero exit (may be ALREADY_EXISTS)" -ForegroundColor Yellow
    }
}

Write-Host "==> Setting project to $ProjectId"
gcloud config set project $ProjectId
if ($LASTEXITCODE -ne 0) { Write-Host "Failed to set project" -ForegroundColor Red; exit 1 }

# ---- Enable APIs ----
Write-Host "==> Enabling required APIs..."
gcloud services enable `
    run.googleapis.com `
    artifactregistry.googleapis.com `
    storage.googleapis.com `
    iam.googleapis.com `
    iamcredentials.googleapis.com `
    cloudbuild.googleapis.com
if ($LASTEXITCODE -ne 0) { Write-Host "Failed to enable APIs" -ForegroundColor Red; exit 1 }

# ---- Artifact Registry ----
Write-Host "==> Creating Artifact Registry repository..."
$output = gcloud artifacts repositories create $AppName `
    --repository-format=docker `
    --location=$Region `
    --description="Docker images for $AppName" 2>&1
if ($LASTEXITCODE -ne 0) {
    if ("$output" -match "ALREADY_EXISTS|already exists|conflict") { Write-Host "    (already exists)" }
    else { Write-Host "    FAILED: $output" -ForegroundColor Red; exit 1 }
}

# ---- GCS Bucket for models ----
$Bucket = "$ProjectId-$AppName-models"
Write-Host "==> Creating GCS bucket gs://$Bucket ..."
$output = gcloud storage buckets create "gs://$Bucket" `
    --location=$Region `
    --uniform-bucket-level-access 2>&1
if ($LASTEXITCODE -ne 0) {
    if ("$output" -match "already exists|ALREADY_EXISTS|409") { Write-Host "    (already exists)" }
    else { Write-Host "    FAILED: $output" -ForegroundColor Red; exit 1 }
}

# ---- Backend Service Account (reads model weights) ----
$BackendSA = "$AppName-backend@$ProjectId.iam.gserviceaccount.com"
Write-Host "==> Creating backend service account..."
$output = gcloud iam service-accounts create "$AppName-backend" `
    --display-name="$AppName backend Cloud Run SA" 2>&1
if ($LASTEXITCODE -ne 0) {
    if ("$output" -match "ALREADY_EXISTS|already exists|conflict") { Write-Host "    (already exists)" }
    else { Write-Host "    FAILED: $output" -ForegroundColor Red; exit 1 }
}

# Wait for SA to propagate before binding IAM
Write-Host "    Waiting for service account to propagate..."
Start-Sleep -Seconds 10

gcloud storage buckets add-iam-policy-binding "gs://$Bucket" `
    --member="serviceAccount:$BackendSA" `
    --role="roles/storage.objectViewer" `
    --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host "    WARNING: bucket IAM binding may have failed" -ForegroundColor Yellow }

# ---- CI/CD Service Account ----
$CiSA = "$AppName-ci@$ProjectId.iam.gserviceaccount.com"
Write-Host "==> Creating CI/CD service account..."
$output = gcloud iam service-accounts create "$AppName-ci" `
    --display-name="$AppName CI/CD service account" 2>&1
if ($LASTEXITCODE -ne 0) {
    if ("$output" -match "ALREADY_EXISTS|already exists|conflict") { Write-Host "    (already exists)" }
    else { Write-Host "    FAILED: $output" -ForegroundColor Red; exit 1 }
}

# Wait for SA to propagate
Write-Host "    Waiting for service account to propagate..."
Start-Sleep -Seconds 10

foreach ($Role in @("roles/run.admin", "roles/artifactregistry.writer", "roles/iam.serviceAccountUser")) {
    gcloud projects add-iam-policy-binding $ProjectId `
        --member="serviceAccount:$CiSA" `
        --role=$Role `
        --quiet 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host "    WARNING: failed to grant $Role" -ForegroundColor Yellow }
}
Write-Host "    Granted: run.admin, artifactregistry.writer, iam.serviceAccountUser"

# ---- Workload Identity Federation ----
$PoolName = "$AppName-github"
$ProviderName = "github"

Write-Host "==> Creating Workload Identity Pool..."
$output = gcloud iam workload-identity-pools create $PoolName `
    --location="global" `
    --display-name="GitHub Actions pool" 2>&1
if ($LASTEXITCODE -ne 0) {
    if ("$output" -match "ALREADY_EXISTS|already exists|conflict") { Write-Host "    (already exists)" }
    else { Write-Host "    FAILED: $output" -ForegroundColor Red; exit 1 }
}

Write-Host "==> Creating Workload Identity Provider..."
$condition = 'assertion.repository == \"' + $GitHubRepo + '\"'
$output = gcloud iam workload-identity-pools providers create-oidc $ProviderName `
    --location="global" `
    --workload-identity-pool=$PoolName `
    --display-name="GitHub OIDC" `
    --issuer-uri="https://token.actions.githubusercontent.com" `
    --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" `
    --attribute-condition=$condition 2>&1
if ($LASTEXITCODE -ne 0) {
    if ("$output" -match "ALREADY_EXISTS|already exists|conflict") { Write-Host "    (already exists)" }
    else { Write-Host "    FAILED: $output" -ForegroundColor Red; exit 1 }
}

# Allow GitHub Actions WIF to impersonate the CI service account
$ProjectNumber = gcloud projects describe $ProjectId --format="value(projectNumber)"
$WifMember = "principalSet://iam.googleapis.com/projects/$ProjectNumber/locations/global/workloadIdentityPools/$PoolName/attribute.repository/$GitHubRepo"

gcloud iam service-accounts add-iam-policy-binding $CiSA `
    --role="roles/iam.workloadIdentityUser" `
    --member=$WifMember `
    --quiet 2>&1 | Out-Null

# ---- Print outputs ----
$WifProvider = "projects/$ProjectNumber/locations/global/workloadIdentityPools/$PoolName/providers/$ProviderName"

Write-Host ""
Write-Host "============================================="
Write-Host " Bootstrap complete!"
Write-Host "============================================="
Write-Host ""
Write-Host "Add these secrets to your GitHub repo:"
Write-Host "  Settings > Secrets and variables > Actions > New repository secret"
Write-Host ""
Write-Host "  GCP_PROJECT_ID      = $ProjectId"
Write-Host "  GCP_WIF_PROVIDER    = $WifProvider"
Write-Host "  GCP_SERVICE_ACCOUNT = $CiSA"
Write-Host ""
Write-Host "Upload your model files:"
Write-Host "  gsutil cp models/*.pth gs://$Bucket/models/"
Write-Host ""
Write-Host "Then push to master and GitHub Actions will deploy automatically."
