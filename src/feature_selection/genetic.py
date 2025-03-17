import random
from collections import defaultdict
import torch
from tqdm.auto import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from torchvision import transforms

from ..training.train import train_model
from ..preprocessing.dataset import AudioDataset
from ..models import BaseCNNModel
from ..utils.config import config

ga_config = config['genetic_algorithm']
POPULATION_SIZE = ga_config['population_size']
NUM_GENERATIONS = ga_config['num_generations']
MUTATION_RATE = ga_config['mutation_rate']
TARGET_FEATURE_COUNT = ga_config['target_feature_count']
NEW_IND_PROB = ga_config['new_individual_prob']

feature_weights = ga_config['feature_weights']
feature_counts = ga_config['feature_counts']

TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor(),
])

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

def generate_individual():
    individual = defaultdict(list)
    selected_features = 0
    while selected_features < TARGET_FEATURE_COUNT:
        feature_type = random.choices(list(feature_counts.keys()), weights=feature_weights.values())[0]
        feature_index = random.randint(0, feature_counts[feature_type] - 1)

        # Add unique features only
        if feature_index not in individual[feature_type]:
            individual[feature_type].append(feature_index)
            selected_features += 1
    return individual

def generate_population():
    return [generate_individual() for _ in range(POPULATION_SIZE)]

def mutate(individual):
    for i in range(TARGET_FEATURE_COUNT):
        if random.random() < MUTATION_RATE:
            feature_type = random.choices(list(feature_counts.keys()), 
                                       weights=feature_weights.values())[0]
            index = random.randint(0, len(individual[feature_type]) - 1)
            feature_index = random.randint(0, feature_counts[feature_type] - 1)

            while feature_index not in individual[feature_type]:
                individual[feature_type][index] = feature_index
                feature_index = random.randint(0, feature_counts[feature_type] - 1)
                
    return individual

def crossover(parent1, parent2):
    child1 = defaultdict(list)
    child2 = defaultdict(list)

    if random.random() <= NEW_IND_PROB:
        child1 = generate_individual()
        child2 = generate_individual()
    else:
        combined_features = []
        for feature in parent1.keys():
            combined_features.extend([(feature, f) for f in parent1[feature]])

        for feature in parent2.keys():
            combined_features.extend([(feature, f) for f in parent2[feature]])

        combined_features = list(set(combined_features))
        selected_features1 = random.choices(combined_features, k=TARGET_FEATURE_COUNT)
        selected_features2 = random.choices(combined_features, k=TARGET_FEATURE_COUNT)

        for feature, index in selected_features1:
           child1[feature].append(index)

        for feature, index in selected_features2:
           child2[feature].append(index)

    return [child1, child2]


def fitness(individual, dataframes, config, display_disabled=True):

    rng = torch.Generator().manual_seed(config['seed'])

    datasets = {TRAIN : AudioDataset(dataframes[TRAIN], config['data']['data_folder'], transform=TRANSFORM, selected_features=individual),
                VAL : AudioDataset(dataframes[VAL], config['data']['data_folder'], transform=TRANSFORM, selected_features=individual),
                TEST : AudioDataset(dataframes[TEST], config['data']['data_folder'], transform=TRANSFORM, selected_features=individual)}

    dataloaders = {TRAIN: DataLoader(datasets[TRAIN], config['model']['train_batch_size'], shuffle=True, generator=rng),
                  VAL: DataLoader(datasets[VAL], config['model']['validation_batch_size'], shuffle=False, generator=rng),
                  TEST: DataLoader(datasets[TEST], config['model']['test_batch_size'], shuffle=False, generator=rng)
                  }

    # Train and evaluate the model (example function)
    model = BaseCNNModel().to(config['device'])

    criterion = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=config['model']['learning_rate'], weight_decay=0.0002)

    model, score = train_model(model, criterion, optimizer, dataloaders, num_epochs = 10, display_disabled=display_disabled)

    fitness_score = 1 / (1 + score)


    return fitness_score

def genetic_algorithm(dataframes):
    population = generate_population()

    for generation in tqdm(range(NUM_GENERATIONS), total=NUM_GENERATIONS, position=0, leave=True):
        fitness_scores = [(individual, fitness(individual, dataframes)) 
                         for individual in tqdm(population, total=len(population), position=1)]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        fitness_scores_df = pd.DataFrame(fitness_scores, columns=['features', 'fitness_score'])
        fitness_scores_df.to_csv(f'Data/fitness_scores_gen{generation}.csv', index=False)

        print(f'Top 5 fitness scores generation {generation}: \n', fitness_scores_df[:5])

        selected_parents = random.choices(population, 
                                       weights=fitness_scores_df['fitness_score'], 
                                       k=POPULATION_SIZE)

        new_population = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1, parent2 = selected_parents[i], selected_parents[i + 1]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population

    return fitness_scores_df
