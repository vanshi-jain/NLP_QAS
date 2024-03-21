from transformers import AutoTokenizer, AutoModel,  DistilBertForQuestionAnswering
import torch

# edit this
context = "The US has passed the peak on new coronavirus cases, " \
          "President Donald Trump said and predicted that some states would reopen this month. " \
          "The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world."

# edit this
question = "What was President Donald Trump's prediction?"

# edit this
model_path = "/home/vanam.b/NLP/QA_output"


model = AutoModel.from_pretrained(model_path)
model = DistilBertForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


encoding = tokenizer.encode_plus(question, context)

input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

output = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))

start_scores = output['start_logits']
end_scores = output['end_logits']

ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)


print ("\nQuestion ",question)
print ("\nAnswer Tokens: ")
print (answer_tokens)

answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)

print ("\nAnswer : ",answer_tokens_to_string)




