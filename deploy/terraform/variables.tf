# Carrega variáveis do arquivo .env no diretório raiz
locals {
  # 1. Lê o arquivo e divide por quebras de linha
  env_lines = split("\n", file("../../.env"))

  # 2. Filtra apenas linhas válidas (ignora vazias e comentários com #)
  valid_lines = [
    for line in local.env_lines : trimspace(line)
    if trimspace(line) != "" && !startswith(trimspace(line), "#")
  ]

  # 3. Transforma em um mapa chave => valor limpando espaços extras
  env_vars = {
    for line in local.valid_lines :
    # Divide a linha no primeiro '=' encontrado e limpa os espaços da chave
    trimspace(split("=", line)[0]) => sensitive(
      # Limpa os espaços do valor e remove as aspas duplas iniciais/finais se existirem
      trim(trimspace(substr(line, length(split("=", line)[0]) + 1, -1)), "\"")
    )
  }
}