# Why am I doing this?

Have been giving interviews lately and focus in the past few weeks have been in agentic pipelines and core systems work.

This made me lose a lot of my ML/DL muscle that I would like to get back on-track on.

What's better than just re-implementing GPT from scratch on a small text dataset of sorts? :)

Going to implement more DL stuff and get back on-track with learning math!

# Overall flow

The overall flow will be as follows:

- loading the dataset
- splitting the dataset - simple splits, not going to worry about cross-validation and all that shiiiit.
- implementing a very simple barebones tokenizer (dictionary)
- implementing the layers - only forward pass, not going to do backward passes, that would be torture.
- writing the forward pass
- writing the training loops
- training the model

extra steps:

- compiling the model for production - getting to barebones model and stripping out the runtime.
- writing a simple server to send inference requests to, perhaps in golang.
