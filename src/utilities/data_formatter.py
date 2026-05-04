"""
Data Formatter Utility

Handles dynamic formatting of training data and prompts for any model
using Hugging Face's tokenizer chat templates.
"""


class DataFormatter:
    """Data formatting utility (SRP: single responsibility for data formatting)"""
    
    @staticmethod
    def format_training_data(example: list, tokenizer=None) -> str:
        """
        Format training example dynamically based on the model's native template.
        Works for Qwen (ChatML), Gemma 4 (<start_of_turn>), etc.
        """
        # Best Practice: Let the tokenizer handle the exact syntax
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                example, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
        # Fallback to Qwen ChatML if tokenizer is missing (legacy support)
        formatted = ""
        for msg in example:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
        return formatted
    
    @staticmethod
    def format_prompt(prompt: str, tokenizer=None) -> str:
        """Format a prompt for model inference dynamically"""
        messages = [{"role": "user", "content": prompt}]
        
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
        # Fallback
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
