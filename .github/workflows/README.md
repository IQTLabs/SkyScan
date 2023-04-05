# SkyScan GitHub Workflows

## Overview

SkyScan and associated projects use a number of GitHub action workflows to verify code security, secrets integrity, and provide project building and deployment capability. These workflows and their required secrets and configuration variables are detailed below. 

For information on adding configuration variables and secrets, see the GitHub Actions documentation [here](https://docs.github.com/en/actions/learn-github-actions/variables#creating-configuration-variables-for-an-environment) and [here](https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-a-repository).

## Docker Container Building 

SkyScan uses the action outlined in `dockerbuild.yml` to automatically build Dockerfiles located in the project and push them to Docker Hub. This action is triggered everytime there is a push to the `main` branch. 

### Secrets
* `DOCKER_USERNAME` (required) - The login username for Docker Hub. 
* `DOCKER_NAMESPACE` (required) - The Docker Hub account that this image will be pushed to. The `DOCKER_USERNAME` Docker Hub user must have write permissions to `DOCKER_NAMESPACE`. Oftentimes they are the same value. 
* `DOCKER_TOKEN` (required) - A Docker Hub personal access token associated with the `DOCKER_USERNAME` account. 

### Configuration Variables
* `PROJECT_NAME` (required) - The overarching name of the project. 
* `DOCKER_BUILD_PLATFORMS` (optional, defaults to `linux/amd64`) - A comma separated list of the target platforms the images will be built for. 
* `DOCKER_BUILD_FOLDERS` (optional, defaults to `.`) - A comma separated list of folders under the main repository containing Dockerfiles to be built. 