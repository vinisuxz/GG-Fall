# GG Fall (protótipo inicial)

Jogo homebrew 3D para Xbox 360, inspirado no gênero party-platformer, com identidade própria.

## Estado real deste pacote

Este é o **esqueleto inicial do projeto**, não um jogo completo. Ele configura o fluxo de CI e inclui uma cena 3D de validação baseada no sample `cube` do SDK xsdk. O primeiro artefato gerado será `GG-Fall.xex` para testar a toolchain e a execução. A jogabilidade completa (personagem low-poly, física, obstáculos e câmera de terceira pessoa) ainda precisa ser implementada sobre a camada de renderização do SDK.

## Requisitos importantes

- O SDK [xenosworks/xsdk](https://github.com/xenosworks/xsdk) produz XEX para o kernel retail em consoles **RGH/JTAG**; não presuma compatibilidade com todo exploit de software.
- ABadAvatar é um ponto de entrada do exploit Bad Update, não um substituto permanente de RGH. A execução depende do payload pós-exploit que você usa e do estado atual do console.
- O workflow do GitHub Actions compila na nuvem. A primeira execução pode demorar bastante porque constrói o toolchain PowerPC e as ferramentas de shader.
- Este projeto não inclui arquivos proprietários, chaves, firmware ou SDK vazado.

## Como usar pelo celular

1. Baixe e extraia o ZIP.
2. No GitHub, crie um repositório chamado `GG-Fall`.
3. Envie todos os arquivos e pastas deste pacote para a raiz do repositório (incluindo `.github/workflows/build.yml`).
4. Abra a aba **Actions** do repositório e habilite workflows, se solicitado.
5. Abra o workflow **Build GG Fall (Xbox 360)** e toque em **Run workflow**.
6. Quando terminar com sucesso, abra a execução, vá em **Artifacts** e baixe `GG-Fall-xex`.
7. Extraia o ZIP do artefato. Ele deve conter `GG-Fall.xex`.

## Primeiro build

O workflow:
1. baixa xsdk;
2. constrói o compilador `powerpc-xsdk-elf`;
3. constrói o compilador de shaders compatível;
4. compila o sample 3D `cube` do xsdk;
5. empacota o XEX como `GG-Fall.xex` para validar o pipeline.

**Honestidade técnica:** este primeiro XEX ainda é o cubo de demonstração do SDK com o nome do projeto; não é a versão final do GG Fall. A próxima etapa é substituir a cena de teste por geometria de personagem, plataformas, obstáculos e lógica de jogo. Não foi possível validar a execução no seu Xbox a partir daqui.

## Próximas etapas de desenvolvimento

- [ ] Personagem low-poly e animação simples
- [ ] Movimento com analógico esquerdo e pulo no botão A
- [ ] Câmera em terceira pessoa com controle pelo analógico direito
- [ ] Pista flutuante, plataformas, obstáculos móveis e checkpoints
- [ ] Tela de início, pausa e reinício
- [ ] Teste no console e otimização para 720p
