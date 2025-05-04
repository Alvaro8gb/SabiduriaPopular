from embeddings.evaluate import similarity_with_cohere
from datasets import load_dataset
import os, pathlib

current_file_path = pathlib.Path(os.path.abspath(__file__))
data_folder = os.path.join(current_file_path.parent,"data")
output_folder = os.path.join(current_file_path.parent,"output")

ds_llama2 = load_dataset("csv", data_files=[os.path.join(data_folder,"benchmark_responses_llama-2-7b.tsv")], sep="\t", split="train")
similarities, mean = similarity_with_cohere(ds_llama2["Respuesta"],ds_llama2["Model_output"])
ds_llama2 = ds_llama2.add_column(name="SimilarityCohere",column=similarities)
ds_llama2.to_csv(os.path.join(output_folder,"eval-benchmark_responses_llama-2-7b.tsv"), sep="\t")
print(f"Llama2 Mean similarity {mean}")

ds_llama3 = load_dataset("csv", data_files=[os.path.join(data_folder,"benchmark_responses_llama-3-7b.tsv")], sep="\t", split="train")
similarities, mean = similarity_with_cohere(ds_llama3["Respuesta"],ds_llama3["Model_output"])
ds_llama3 = ds_llama3.add_column(name="SimilarityCohere",column=similarities)
ds_llama3.to_csv(os.path.join(output_folder,"eval-benchmark_responses_llama-3-7b.tsv"), sep="\t")
print(f"Llama3 Mean similarity {mean}")

ds_mistral = load_dataset("csv", data_files=[os.path.join(data_folder,"benchmark_responses_mistral-7b-instruct-v0.3.tsv")], sep="\t", split="train")
similarities, mean = similarity_with_cohere(ds_mistral["Respuesta"],ds_mistral["Model_output"])
ds_mistral = ds_mistral.add_column(name="SimilarityCohere",column=similarities)
ds_mistral.to_csv(os.path.join(output_folder,"benchmark_responses_mistral-7b-instruct-v0.3.tsv"), sep="\t")
print(f"Mistral Mean similarity {mean}")
