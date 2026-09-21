# Ensaio — API de análise

Backend que recebe um vídeo e devolve transcrição + linguagem corporal
(contato visual, olhar, postura, sorriso) num JSON só.

## Testar localmente (opcional, exige Python 3.11 e ~5GB de espaço para os modelos)

```
cd backend
pip install -r requirements.txt
mkdir models
# baixar os 3 arquivos .task nos links que estão no Dockerfile, salvar em models/
uvicorn app.main:app --reload --port 8000
```

Depois abrir `http://localhost:8000/docs` — o FastAPI gera uma página de
teste automática onde dá pra subir um vídeo e ver o resultado, sem
escrever nenhum código.

## Publicar no Render (recomendado, gratuito)

1. Suba essa pasta inteira (`backend/`) para um repositório no GitHub —
   pode ser um repositório novo, privado se preferir.
2. Em [render.com](https://render.com), clique em **New +** → **Web Service**.
3. Conecte sua conta do GitHub e escolha esse repositório.
4. Em **Root Directory**, coloque `backend` (se o repositório tiver outras
   pastas além dela) ou deixe em branco se `backend/` for a raiz do repo.
5. Em **Runtime**, escolha **Docker** (o Render detecta o `Dockerfile`
   sozinho, não precisa configurar comando de start).
6. Em **Instance Type**, escolha **Free**.
7. Clique em **Create Web Service**.

O primeiro build demora — uns 10 a 15 minutos, porque ele baixa e instala
o MediaPipe, o Whisper e os modelos. As próximas vezes que você atualizar
o código, é mais rápido.

Quando terminar, o Render te dá uma URL tipo
`https://ensaio-api-xxxx.onrender.com`. É esse endereço que o frontend vai
chamar.

## Testar se está no ar

Abra `https://sua-url.onrender.com/docs` no navegador — se aparecer a
página de documentação do FastAPI, está funcionando.

## Sobre o plano gratuito do Render

- O serviço "dorme" depois de um tempo sem uso. A primeira chamada depois
  disso demora uns 30-50 segundos para acordar — normal, não é erro.
- Processar um vídeo de alguns minutos pode levar de 1 a 4 minutos,
  dependendo do tamanho — CPU do plano free não é rápida. Se ficar lento
  demais na prática, é o sinal de que vale considerar um plano pago.
- Limite de upload configurado em 500MB (em `app/main.py`, constante
  `MAX_UPLOAD_MB`) — pode reduzir se quiser respostas mais rápidas.
