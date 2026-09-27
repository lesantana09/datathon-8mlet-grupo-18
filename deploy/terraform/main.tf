# 1. Configuração do Provedor AWS
provider "aws" {
  region     = "us-east-2"
  access_key = local.env_vars.AWS_ACCESS_KEY
  secret_key = local.env_vars.AWS_SECRET_KEY
}

# 2. Infraestrutura de Rede Básica (VPC e Subnets)
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
}

resource "aws_subnet" "public_1" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "us-east-2a"
  map_public_ip_on_launch = true
}

resource "aws_subnet" "public_2" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = "us-east-2b"
  map_public_ip_on_launch = true
}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id
}

resource "aws_route_table" "rt" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gw.id
  }
}

resource "aws_route_table_association" "a" {
  subnet_id      = aws_subnet.public_1.id
  route_table_id = aws_route_table.rt.id
}

resource "aws_route_table_association" "b" {
  subnet_id      = aws_subnet.public_2.id
  route_table_id = aws_route_table.rt.id
}

# 3. Security Group (Controle de Acesso - Liberada porta 8081)
resource "aws_security_group" "ecs_sg" {
  name        = "ecs-fargate-sg"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 8081
    to_port     = 8081
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # Permite acesso de qualquer IP à sua aplicação
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"] # Obrigatório para o Fargate baixar a imagem na internet
  }
}

# 4. AWS Secrets Manager (Guarda as credenciais do Docker Hub com segurança)
resource "aws_secretsmanager_secret" "docker_hub" {
  name                    = "docker-hub-credentials"
  recovery_window_in_days = 0 # Permite deletar imediatamente sem reter por 30 dias se necessário
}

resource "aws_secretsmanager_secret_version" "docker_hub_val" {
  secret_id     = aws_secretsmanager_secret.docker_hub.id
  secret_string = jsonencode({
    username = local.env_vars.DOCKER_HUB_USERNAME
    password = local.env_vars.DOCKER_HUB_TOKEN
  })
}

# 5. IAM Roles (Permissões de Execução do ECS e Acesso ao Secret)
resource "aws_iam_role" "ecs_execution_role" {
  name = "ecs_task_execution_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

# Permissão padrão do ECS
resource "aws_iam_role_policy_attachment" "ecs_execution_attach" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Permissão customizada para o ECS conseguir ler o segredo do Docker Hub no Secrets Manager
resource "aws_iam_policy" "ecs_secret_policy" {
  name        = "ecs_secret_read_policy"
  description = "Permite que o ECS leia o token do Docker Hub"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [aws_secretsmanager_secret.docker_hub.arn]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_secret_attach" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = aws_iam_policy.ecs_secret_policy.arn
}

# 6. Criando o Cluster ECS
resource "aws_ecs_cluster" "main" {
  name = "datathon-cluster"
}

# 7. Task Definition (Configuração do Contêiner com Credenciais e Porta 8081)
resource "aws_ecs_task_definition" "app" {
  family                   = "datathon-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn

  container_definitions = jsonencode([{
    name      = "datathon-container"
    image     = "ghcr.io/tramontano/datathon-8mlet-grupo-18:1.0.0"
    essential = true
    command   = ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8081"]
    environment = [
      { name = "API_HOST",               value = local.env_vars.API_HOST },
      { name = "API_PORT",               value = local.env_vars.API_PORT },
      { name = "API_V1_STR",             value = local.env_vars.API_V1_STR },
      { name = "WEB_CONCURRENCY",         value = local.env_vars.WEB_CONCURRENCY },
      { name = "API_USERNAME",           value = local.env_vars.API_USERNAME },
      { name = "API_PASSWORD",           value = local.env_vars.API_PASSWORD },
      { name = "DAGSHUB_APP_TOKEN",       value = local.env_vars.DAGSHUB_APP_TOKEN },
      { name = "MLFLOW_TRACKING_URI",    value = local.env_vars.MLFLOW_TRACKING_URI },
      { name = "MLFLOW_EXPERIMENT_NAME", value = local.env_vars.MLFLOW_EXPERIMENT_NAME },
      { name = "MLFLOW_TRACKING_USERNAME", value = local.env_vars.MLFLOW_TRACKING_USERNAME },
      { name = "BUCKET_NAME",            value = local.env_vars.BUCKET_NAME },
      { name = "AWS_REGION",             value = local.env_vars.AWS_REGION },
      { name = "AWS_ENDPOINT_URL",       value = local.env_vars.AWS_ENDPOINT_URL },
      { name = "AWS_ACCESS_KEY",         value = local.env_vars.AWS_ACCESS_KEY },
      { name = "AWS_SECRET_KEY",         value = local.env_vars.AWS_SECRET_KEY }
    ]


    portMappings = [{
      containerPort = 8081
      hostPort      = 8081
    }]
  }])
}

# 8. ECS Service (Mantém as instâncias rodando)
resource "aws_ecs_service" "main" {
  name            = "datathon-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  # Otimização de Custo: Usando Fargate Spot (60-70% mais barato)
  #capacity_provider_strategy {
  #  capacity_provider = "FARGATE_SPOT"
  #  weight            = 1
  #}

  network_configuration {
    subnets          = [aws_subnet.public_1.id, aws_subnet.public_2.id]
    security_groups  = [aws_security_group.ecs_sg.id]
    assign_public_ip = true
  }
}

output "cluster_name" {
  description = "Nome do Cluster ECS criado"
  value       = aws_ecs_cluster.main.name
}

output "service_name" {
  description = "Nome do Serviço ECS"
  value       = aws_ecs_service.main.name
}

output "get_public_ip" {
  description = "Obtém o IP público atual do container"
  value       = <<EOF
aws ecs list-tasks --cluster \({aws_ecs_cluster.main.name} --query 'taskArns[0]' --output text \vert{} xargs -I {} aws ecs describe-tasks --cluster\){aws_ecs_cluster.main.name} --tasks {} --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text | xargs -I {} aws ec2 describe-network-interfaces --network-interface-ids {} --query 'NetworkInterfaces[0].Association.PublicIp' --output text
EOF
}
