# Choosing a model

AI Chat works only with **instruction-tuned** models. The model is whichever one
llama-server is running; the app never chooses it. Whether a model you have downloaded is
instruction-tuned is for you to determine.

## Base models and instruction-tuned models

A **base model** is trained only to continue text. Given some words, it predicts what
comes next. It has no concept of a turn, of a user, or of when to stop.

An **instruction-tuned** model (often called "instruct" or "chat") starts from a base model
and is trained further on conversations written in a specific turn format, called its
*chat template*. It learns to read a user's message, write one answer, and stop.

AI Chat has the server apply the model's chat template (`/apply-template`, then
`/completion`), so the model receives the turn format it was trained on.

## Why a base model is not useful here

- **No template, no stopping point.** A base model has no chat template of its own, so
  llama-server falls back to a generic one (ChatML) that the model was never trained on.
  The model does not know where its reply should end, and tends to run on, often writing
  the user's next message as well.
- **Imitating a conversation is fragile.** A plain `User:` / `Assistant:` transcript with a
  stop string can work for a single question. In a prototype on one model it was
  unreliable over several turns: the model refused, asked the question back, or repeated
  itself. That was not a measured result.
- **The app assumes turns.** The system prompt, replies that end, and retrying or recalling
  whole exchanges all rely on a model that answers once and stops.
- **Chat-like wording proves nothing.** Some base models imitate an assistant and will
  sometimes answer like one.

## What to do

Use an instruction-tuned build of the model. Start llama-server with it, then start the
chat.
