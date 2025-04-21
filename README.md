# PeptideGPT

A GPT-2 based protein language model for designing proteins with specific properties. The models can be found on [Hugging Face](https://huggingface.co/collections/aayush14/peptidegpt-66f9f3efec6983f03c0efdb6). 

## Important Files

- **inference.py**: This script runs inferences using a chosen model.
- **data**: Contains the necessary data for model training and evaluation.

## Files

- **inference.py**: This script is used for running inference on the trained models to generate peptide sequences.

## Usage

1. Clone this repository.
2. Install the required dependencies (`pip install -r requirements.txt`)
3. To generate sequences and run the inference pipeline, use `python inference.py --model_path=path_to_model --num_return_sequences=num --max_length=max_len --starts_with=starting_sequence --output_dir=output_directory --pred_model_path=path_to_prediction_model --seed=random_seed`
   
   * model_path: Path of the model to run generation from. You can choose any of the four models from the models folder. 
   * num_return_sequences: Number of sequences to generate (default is 1000).
   * max_length: Maximum length of generated sequences (default is 50).
   * starts_with: Starting amino acids for generation (default is an empty string).
   * output_dir: Directory for storing all output files.
   * pred_model_path: You need to clone PeptideBERT and give its path to run the predictions. 
   * seed: Random seed for reproducibility (default is 42).
4. Hugging Face's [run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py) script can be used to fine-tune the model on a custom dataset. 


