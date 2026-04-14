# Why am I doing this?

my focus in the past few weeks have been in agentic pipelines and core systems work. I feel this made me lose a lot of my ML/DL muscle that I would like to get back on-track on.

What's better than just re-implementing GPT from scratch on a small text dataset of sorts? :)
Going to implement more DL stuff and get back on-track with learning math!

# Overall flow

paper link: <https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>

changes: word-level tokenizer

The overall flow will be as follows:

- [x] loading the dataset
- [x] implementing a very simple barebones tokenizer (dictionary)
- [x] implementing the layers - only forward pass, not going to do backward passes, that would be torture.
- [x] writing the forward pass
- [ ] splitting the dataset - simple splits, not going to worry about cross-validation and all that shiiiit.
- [x] writing the training loops
- [x] writing inference func
- [ ] training the model
- [ ] implementing kv-cache for inference

extra steps:

- [ ] compiling the model for production - getting to barebones model and stripping out the runtime.
- [ ] writing a simple server to send inference requests to, perhaps in golang.

# Architecture
architecture image for reference

![gpt-2 arch](https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Full_GPT_architecture.svg/500px-Full_GPT_architecture.svg.png)
