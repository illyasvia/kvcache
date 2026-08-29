import os
import sys

sys.path.append("../")

# 仓库根目录，用于定位 weight/ 下的默认权重
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
# from transformers import Qwen2_5OmniProcessor
import torch


class Qwen3vl:
    def __init__(self, model_path = None):
        if model_path is None:
            model_path = os.path.join(REPO_ROOT, "weight", "Qwen", "Qwen3-VL-4B-Instruct")
        self.model_path = model_path
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype=torch.float16, # device_map="auto",
            device_map="auto", tp_plan=None, trust_remote_code=True, attn_implementation="flash_attention_2", )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def get_punctuation(self,input_ids):
        punctuation_chars = ",.!?;:\t\n" 
        punct_token_ids = set()
        for char in punctuation_chars:
            token_id = self.processor.tokenizer.encode(char, add_special_tokens=False)[0]
            punct_token_ids.add(token_id)
        mask = []
        for idx,id in enumerate(input_ids):
            if id.item() in punct_token_ids:
                mask.append(1)
            else:
                mask.append(0)
        return torch.tensor(mask)
            
    def process(self, prompt, instruction = None, talker_max_new_tokens=None, thinker_max_new_tokens=None):
        conversation = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}], }, {"role": "user", "content": [{"type": "text", "text": prompt}], }]

        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        
        mask = self.get_punctuation(inputs.input_ids[0])
        self.model.update_mask(mask)
        
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        text_ids = self.model.generate(**inputs, talker_max_new_tokens=talker_max_new_tokens, thinker_max_new_tokens=thinker_max_new_tokens)
        text = self.processor.batch_decode(text_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return text[0]

    def process_multiple_multimodel(self,prompt, images, instruction = None,role= None,talker_max_new_tokens=None, thinker_max_new_tokens=None):
        if role is None:
            role = "You are a helpful assistant."
        messages = [{"role": "system", "content": [
        {"type": "text", "text": role}], }, 
        {"role": "user", "content": [], }]
        for img in images:
            messages[1]["content"].append({"type": "image","image": img})
        messages[1]["content"].append({"type": "text", "text": prompt})
        if instruction is not None:
             messages[1]["content"].append({"type": "text", "text": instruction})
        inputs = self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt")
        device = next(self.model.parameters()).device  
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "thinking" in self.model_path.lower():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        else:
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        # import ipdb; ipdb.set_trace()
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text

    def process_multimodel(self,prompt, image, instruction = None, role = None,talker_max_new_tokens=None, thinker_max_new_tokens=None):
        if role is None:
            role = "You are a helpful assistant."
        messages = [{"role": "system", "content": [
        {"type": "text", "text": role}], }, 
        {"role": "user", "content": [
            {"type": "image","image": image,},
            {"type": "text", "text": prompt}], }]
        if instruction is not None:
             messages[1]["content"].append({"type": "text", "text": instruction})
        inputs = self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt")
        device = next(self.model.parameters()).device  
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "thinking" in self.model_path.lower():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        else:
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        # import ipdb; ipdb.set_trace()
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text