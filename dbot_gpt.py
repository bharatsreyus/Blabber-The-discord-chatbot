
import os

dbot_token = os.environ.get("Blabber-token")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


import  discord
#print(discord.__version__)

client = discord.Client()

step = 0
chat_history_ids = []

@client.event  #event decorator/wrapper
async def on_ready():
    print("We've logged in as {}".format(client.user))


@client.event
async def on_message(message):
    
    global step
    global chat_history_ids
    
    print(f"{message.channel} {message.author} {message.author.name} {message.content}")
    
    ###
    #for step in range(5):
    if message.content == "-bot reconnect":
        step = 0
        chat_history_ids = []
    
    elif message.content.startswith("-bot"):
        
        input_string = message.content.lstrip("-bot")
    
    # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(input_string + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
        bot_input_ids =  torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    #torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else
    # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
        await message.channel.send("Bot: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
        print("step:",step)
        step += 1
    ###
    '''
    if message.content == "hi there":
        await message.channel.send("Hola!")

    if message.content == "What is your name?":
        await message.channel.send("you can call me whatever")
    '''

client.run(dbot_token)
