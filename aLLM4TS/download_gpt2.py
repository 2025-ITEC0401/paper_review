from transformers import GPT2Model, GPT2Config

model_name = 'gpt2'

save_directory = './hf_models/gpt2'

print("Start to download..")
model = GPT2Model.from_pretrained(model_name)
config = GPT2Config.from_pretrained(model_name)

model.save_pretrained(save_directory)
config.save_pretrained(save_directory)

print("Finish")
