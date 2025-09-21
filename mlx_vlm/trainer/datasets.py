import warnings
import json


class Dataset:
    def __init__(
        self,
        hf_dataset,
        config,
        processor,
        image_processor=None,
        take=None,
        split=None,
        image_resize_shape=None,
    ):
        if split is not None:
            self.dataset = hf_dataset[split]
        else:
            self.dataset = hf_dataset
        if take is not None:
            self.dataset = self.dataset.take(take)
        self.processor = processor
        self.config = config
        self.image_processor = image_processor
        self.image_resize_shape = image_resize_shape
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        from mlx_vlm.utils import prepare_inputs
        
        item = self.dataset[idx]
        
        images = item.get("images", item.get("image", None))

        if images is None or images == "" or images == []:
            images = []
        elif not isinstance(images, list):
            images = [images]

        image_paths = []
        image_data = []
        for img in images:
            if isinstance(img, str):
                image_paths.append(img)
            else:
                image_data.append(img)
        
        conversations = item.get("messages", item.get("conversations"))
        prompts = []
        
        if isinstance(conversations, list) and isinstance(conversations[0], list):
            for conversation in conversations:
                if self.config["model_type"] == "pixtral":
                    conversation = [json.loads(i) for i in conversation]
                    if len(conversations) > 1:
                        warnings.warn(
                            "Pixtral batch processing is not supported yet. Set batch size to 1."
                        )
                
                if "chat_template" in self.processor.__dict__:
                    prompt = self.processor.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=False,
                        num_images=len(images),
                        num_audios=0,
                    )
                else:
                    prompt = self.processor.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=False,
                        num_images=len(images),
                        num_audios=0,
                    )
                prompts.append(prompt)
        
        else:
            if self.config["model_type"] == "pixtral":
                conversations = [json.loads(i) for i in conversations]
            if "chat_template" in self.processor.__dict__:
                prompt = self.processor.apply_chat_template(
                    conversations,
                    tokenize=False,
                    add_generation_prompt=False,
                    num_images=len(images),
                    num_audios=0,
                )
            else:
                prompt = self.processor.tokenizer.apply_chat_template(
                    conversations,
                    tokenize=False,
                    add_generation_prompt=False,
                    num_images=len(images),
                    num_audios=0,
                )
            prompts.append(prompt)
        
        
        inputs = prepare_inputs(
            processor=self.processor,
            images=image_data,
            audio=None,
            prompts=prompts,
            image_token_index=getattr(self.config, "image_token_index", "image_token_id"),
            resize_shape=self.image_resize_shape
        )
        
        return inputs
    


class PreferenceDataset:
    def __init__(
        self,
        hf_dataset,
        config,
        processor,
        image_processor=None,
        take=None,
        split=None,
        image_resize_shape=None,
    ):
        if split is not None:
            self.dataset = hf_dataset[split]
        else:
            self.dataset = hf_dataset
        if take is not None:
            self.dataset = self.dataset.take(take)
        self.processor = processor
        self.config = config
        self.image_processor = image_processor
        self.image_resize_shape = image_resize_shape
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        from mlx_vlm.utils import prepare_inputs
        
        item = self.dataset[idx]
        
        images = item.get("images", item.get("image", None))

        if images is None or images == "" or images == []:
            images = []
        elif not isinstance(images, list):
            images = [images]

        image_paths = []
        image_data = []
        for img in images:
            if isinstance(img, str):
                image_paths.append(img)
            else:
                image_data.append(img)
        
        system = item.get("system", None)
        prompt = item.get("prompt", None)
        chosen = item.get("chosen", None)
        rejected = item.get("rejected", None)

        if chosen is None or rejected is None or prompt is None:
            raise ValueError("Both 'prompt', 'chosen' and 'rejected' must be present in the dataset items.")

        if self.config["model_type"] == "pixtral":
            conversations = [json.loads(i) for i in conversations]
        if "chat_template" in self.processor.__dict__:
            chosen_conversation = self.processor.apply_chat_template(
                [
                    {"role": "system", "content": system} if system is not None else None,
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen},
                ],
                tokenize=False,
                add_generation_prompt=False,
                num_images=len(images),
                num_audios=0,
            )
            rejected_conversation = self.processor.apply_chat_template(
                [
                    {"role": "system", "content": system} if system is not None else None,
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen},
                ],
                tokenize=False,
                add_generation_prompt=False,
                num_images=len(images),
                num_audios=0,
            )
        else:
            chosen_conversation = self.processor.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system} if system is not None else None,
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen},
                ],
                tokenize=False,
                add_generation_prompt=False,
                num_images=len(images),
                num_audios=0,
            )
            rejected_conversation = self.processor.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system} if system is not None else None,
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen},
                ],
                tokenize=False,
                add_generation_prompt=False,
                num_images=len(images),
                num_audios=0,
            )

        chosen_inputs = prepare_inputs(
            processor=self.processor,
            images=image_data,
            audio=None,
            prompts=chosen_conversation,
            image_token_index=getattr(self.config, "image_token_index", "image_token_id"),
            resize_shape=self.image_resize_shape
        )

        rejected_inputs = prepare_inputs(
            processor=self.processor,
            images=image_data,
            audio=None,
            prompts=chosen_conversation,
            image_token_index=getattr(self.config, "image_token_index", "image_token_id"),
            resize_shape=self.image_resize_shape
        )
        
        return  chosen_inputs, rejected_inputs