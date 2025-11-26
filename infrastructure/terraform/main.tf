terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "vokg-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "vokg-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "VOKG"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# VPC and Networking
module "vpc" {
  source = "./modules/vpc"

  environment         = var.environment
  vpc_cidr            = var.vpc_cidr
  availability_zones  = var.availability_zones
  private_subnets     = var.private_subnets
  public_subnets      = var.public_subnets
  database_subnets    = var.database_subnets
}

# EKS Cluster
module "eks" {
  source = "./modules/eks"

  environment         = var.environment
  cluster_name        = "vokg-${var.environment}"
  cluster_version     = var.eks_cluster_version
  vpc_id              = module.vpc.vpc_id
  private_subnet_ids  = module.vpc.private_subnet_ids

  node_groups = {
    general = {
      desired_size = 3
      min_size     = 2
      max_size     = 10
      instance_types = ["t3.xlarge"]
      capacity_type = "ON_DEMAND"
      labels = {
        workload = "general"
      }
    }

    gpu = {
      desired_size = 1
      min_size     = 0
      max_size     = 5
      instance_types = ["g4dn.xlarge"]  # NVIDIA T4 GPU
      capacity_type = "SPOT"
      labels = {
        workload = "gpu"
        gpu      = "true"
        gpu-type = "nvidia-t4"
      }
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NoSchedule"
      }]
    }
  }
}

# RDS PostgreSQL
module "rds" {
  source = "./modules/rds"

  environment             = var.environment
  vpc_id                  = module.vpc.vpc_id
  database_subnet_ids     = module.vpc.database_subnet_ids

  instance_class          = "db.t3.large"
  allocated_storage       = 100
  max_allocated_storage   = 1000
  engine_version          = "15.4"

  database_name           = "vokg"
  master_username         = var.db_master_username

  multi_az                = true
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "mon:04:00-mon:05:00"

  allowed_security_groups = [module.eks.worker_security_group_id]
}

# ElastiCache Redis
module "elasticache" {
  source = "./modules/elasticache"

  environment         = var.environment
  vpc_id              = module.vpc.vpc_id
  subnet_ids          = module.vpc.private_subnet_ids

  node_type           = "cache.r6g.large"
  num_cache_nodes     = 2
  engine_version      = "7.0"

  allowed_security_groups = [module.eks.worker_security_group_id]
}

# S3 Buckets
module "s3" {
  source = "./modules/s3"

  environment = var.environment

  buckets = {
    videos = {
      name = "vokg-${var.environment}-videos"
      lifecycle_rules = [{
        id      = "archive-old-videos"
        enabled = true
        transition = [{
          days          = 90
          storage_class = "GLACIER"
        }]
      }]
    }

    frames = {
      name = "vokg-${var.environment}-frames"
      lifecycle_rules = [{
        id      = "delete-old-frames"
        enabled = true
        expiration = {
          days = 30
        }
      }]
    }

    masks = {
      name = "vokg-${var.environment}-masks"
      lifecycle_rules = [{
        id      = "delete-old-masks"
        enabled = true
        expiration = {
          days = 30
        }
      }]
    }

    models = {
      name = "vokg-${var.environment}-models"
    }
  }
}

# CloudFront CDN
module "cloudfront" {
  source = "./modules/cloudfront"

  environment    = var.environment
  s3_bucket_ids  = [
    module.s3.bucket_ids["videos"],
    module.s3.bucket_ids["frames"]
  ]

  domain_name    = var.domain_name
  certificate_arn = var.acm_certificate_arn
}

# Secrets Manager
resource "aws_secretsmanager_secret" "vokg_secrets" {
  name = "vokg/${var.environment}/secrets"

  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "vokg_secrets" {
  secret_id = aws_secretsmanager_secret.vokg_secrets.id

  secret_string = jsonencode({
    postgres_password = random_password.db_password.result
    redis_password    = random_password.redis_password.result
    secret_key        = random_password.secret_key.result
  })
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "random_password" "redis_password" {
  length  = 32
  special = false
}

resource "random_password" "secret_key" {
  length  = 64
  special = false
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "vokg" {
  name              = "/aws/vokg/${var.environment}"
  retention_in_days = 30
}

# Outputs
output "eks_cluster_endpoint" {
  value       = module.eks.cluster_endpoint
  description = "EKS cluster endpoint"
}

output "eks_cluster_name" {
  value       = module.eks.cluster_name
  description = "EKS cluster name"
}

output "rds_endpoint" {
  value       = module.rds.endpoint
  description = "RDS endpoint"
  sensitive   = true
}

output "redis_endpoint" {
  value       = module.elasticache.endpoint
  description = "Redis endpoint"
  sensitive   = true
}

output "s3_bucket_names" {
  value       = module.s3.bucket_names
  description = "S3 bucket names"
}

output "cloudfront_domain" {
  value       = module.cloudfront.domain_name
  description = "CloudFront distribution domain"
}
