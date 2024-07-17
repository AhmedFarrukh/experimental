import tensorflow as tf
import pathlib
import re
import subprocess
from collections import defaultdict
from statistics import mean
from statistics import stdev
import matplotlib.pyplot as plt
import numpy as np

modelNames = ["MobileNet", "InceptionV3", "ResNet50", "ResNet101" , "ResNet152", "VGG16", "VGG19"]

def parse_benchmark_output(output):
    """
    Parse benchmark output to extract model initialization times, inference timings, and memory footprint.

    :param output: The raw output string from the benchmark.
    :return: A dictionary containing parsed benchmark results.
    """
    results = {}

    # Regular expressions to match the required information
    model_name_pattern = re.compile(r'INFO: Graph: \[(.*)\]')
    init_time_pattern = re.compile(r'INFO: Initialized session in (\d+.\d+)ms.')
    inference_patterns = [
        re.compile(r'INFO: Inference timings in us: Init: (\d+), First inference: (\d+), Warmup \(avg\): (\d+.\d+), Inference \(avg\): (\d+.\d+)'),
        re.compile(r'INFO: Inference timings in us: Init: (\d+), First inference: (\d+), Warmup \(avg\): (\d+), Inference \(avg\): (\d+)'),
        re.compile(r'INFO: Inference timings in us: Init: (\d+), First inference: (\d+), Warmup \(avg\): ([\d.e+]+), Inference \(avg\): (\d+)')
    ]
    memory_pattern = re.compile(r'INFO: Memory footprint delta from the start of the tool \(MB\): init=(\d+.\d+) overall=(\d+.\d+)')

    current_model = None

    for line in output.split('\n'):
        # Match the model name
        model_match = model_name_pattern.search(line)
        if model_match:
            current_model = model_match.group(1).split('/')[-1].split('.')[0]
            results[current_model] = {}
            continue

        # Match the initialization time
        init_match = init_time_pattern.search(line)
        if init_match and current_model:
            results[current_model]['Init Time (ms)'] = float(init_match.group(1))
            continue

        # Match the inference timings
        for pattern in inference_patterns:
            inference_match = pattern.search(line)
            if inference_match and current_model:
                results[current_model]['Inference Timings (us)'] = {
                    'Init': int(inference_match.group(1)),
                    'First Inference': int(inference_match.group(2)),
                    'Warmup (avg)': float(inference_match.group(3)),
                    'Inference (avg)': float(inference_match.group(4))
                }
                break

        # Match the memory footprint
        memory_match = memory_pattern.search(line)
        if memory_match and current_model:
            results[current_model]['Memory Footprint (MB)'] = {
                'Init': float(memory_match.group(1)),
                'Overall': float(memory_match.group(2))
            }

    return results

results = defaultdict(list)

numModels = len(modelNames)
allModels = modelNames.copy()
for i in range(numModels):
  allModels.append(modelNames[i] + "_quant")

n = 10

for modelName in allModels:
  print(modelName)
  init_time = []
  init_inference = []
  first_inference = []
  warmup_inference = []
  inference = []
  memory_init = []
  memory_overall = []
  for i in range(n):
    outputOriginal = subprocess.check_output("./benchmark/linux_x86-64_benchmark_model \
      --graph=./tflite_models/" + modelName +".tflite"+" \
      --num_threads=1", shell=True)
    outputOriginal = outputOriginal.decode('utf-8')
    output = parse_benchmark_output(outputOriginal)
    try:
      init_time.append(output[modelName]['Init Time (ms)'])
      init_inference.append(output[modelName]['Inference Timings (us)']['Init'])
      first_inference.append(output[modelName]['Inference Timings (us)']['First Inference'])
      warmup_inference.append(output[modelName]['Inference Timings (us)']['Warmup (avg)'])
      inference.append(output[modelName]['Inference Timings (us)']['Inference (avg)'])
      memory_init.append(output[modelName]['Memory Footprint (MB)']['Init'])
      memory_overall.append(output[modelName]['Memory Footprint (MB)']['Overall'])
    except: #error in parsing
      print("Error with model: ", modelName)
      print(output)
      print(outputOriginal)
      continue

  results["Init Time"].append((mean(init_time), stdev(init_time)))
  results["Init Inference"].append((mean(init_inference), stdev(init_inference)))
  results["First Inference"].append((mean(first_inference), stdev(first_inference)))
  results["Warmup Inference"].append((mean(warmup_inference), stdev(warmup_inference)))
  results["Avg Inference"].append((mean(inference), stdev(inference)))
  results["Memory Init"].append((mean(memory_init), stdev(memory_init)))
  results["Memory Overall"].append((mean(memory_overall), stdev(memory_overall)))

for i in range(len(allModels)):
    print(allModels[i])
    for key in results:
        print(key, ": ", results[key][i])
    print('\n')

for key in results:
    means = [x[0] for x in results[key]]
    errors = [x[1] for x in results[key]]

    n_groups = len(modelNames)
    index = np.arange(n_groups)

    fig, ax = plt.subplots()
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, means[:n_groups], bar_width,
                     alpha=opacity,
                     yerr=errors[:n_groups],
                     label=f'{key} (Original)')

    rects2 = plt.bar(index + bar_width, means[n_groups:], bar_width,
                     alpha=opacity,
                     yerr=errors[n_groups:],
                     label=f'{key} (Quantized)')

    plt.xlabel('Model')
    plt.ylabel('Values')
    plt.title(f'Bar Chart for {key}')
    plt.xticks(index + bar_width / 2, modelNames, rotation=45)
    plt.legend()

    plt.tight_layout()

    # Show the plot
    plt.show()