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

ELITE_SIZE = int(POPULATION_SIZE * 0.1)  # Keep top 10% of individuals
TOURNAMENT_SIZE = 5  # Number of individuals in each tournament

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

# def mutate(individual):
#     for i in range(TARGET_FEATURE_COUNT):
#         if random.random() < MUTATION_RATE:
#             feature_type = random.choices(list(feature_counts.keys()), 
#                                        weights=feature_weights.values())[0]
#             index = random.randint(0, len(individual[feature_type]) - 1)
#             feature_index = random.randint(0, feature_counts[feature_type] - 1)

#             while feature_index not in individual[feature_type]:
#                 individual[feature_type][index] = feature_index
#                 feature_index = random.randint(0, feature_counts[feature_type] - 1)
                
#     return individual

def mutate(individual):
    for i in range(TARGET_FEATURE_COUNT):
        if random.random() < MUTATION_RATE:
            # Get feature types that have room for mutation (more features available than used)
            mutable_types = [ft for ft in feature_counts.keys() 
                           if len(individual[ft]) > 0 and  # Has features selected
                           feature_counts[ft] > 1]  # Has more than 1 possible feature
            
            if not mutable_types:
                continue
                
            # Select random feature type from mutable ones
            feature_weights_subset = {k: feature_weights[k] for k in mutable_types}
            weights = [feature_weights_subset[k] for k in mutable_types]
            feature_type = random.choices(mutable_types, weights=weights)[0]
            
            # Select random index to mutate
            index = random.randint(0, len(individual[feature_type]) - 1)
            current_feature = individual[feature_type][index]
            
            # Get all possible features except the current one
            available_features = [f for f in range(feature_counts[feature_type]) 
                                if f != current_feature and f not in individual[feature_type]]
            
            if available_features:  # Only mutate if we have alternative features
                individual[feature_type][index] = random.choice(available_features)
                
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

        # combined_features = list(set(combined_features))
        # selected_features1 = random.choices(combined_features, k=TARGET_FEATURE_COUNT)
        # selected_features2 = random.choices(combined_features, k=TARGET_FEATURE_COUNT)
        
        ### New code 
        # Remove duplicates while maintaining order
        combined_features = list(dict.fromkeys(combined_features))
        
        # Randomly select features ensuring no duplicates per feature type
        for i in range(TARGET_FEATURE_COUNT):
            # Filter available features that haven't been used in child1
            available1 = [(ft, f) for ft, f in combined_features 
                         if f not in child1[ft]]
            if available1:
                ft, f = random.choice(available1)
                child1[ft].append(f)
            
            # Same for child2
            available2 = [(ft, f) for ft, f in combined_features 
                         if f not in child2[ft]]
            if available2:
                ft, f = random.choice(available2)
                child2[ft].append(f)
        ### New code end
        
        # for feature, index in selected_features1:
        #    child1[feature].append(index)

        # for feature, index in selected_features2:
        #    child2[feature].append(index)

        ### New code end
        
        # for feature, index in selected_features1:
        #    child1[feature].append(index)

        # for feature, index in selected_features2:
        #    child2[feature].append(index)

    return [child1, child2]


def fitness(individual, dataframes, config, display_disabled=True):

    rng = torch.Generator().manual_seed(config['seed'])

    datasets = {TRAIN : AudioDataset(dataframes[TRAIN], transform=TRANSFORM, selected_features=individual),
                VAL : AudioDataset(dataframes[VAL], transform=TRANSFORM, selected_features=individual),
                TEST : AudioDataset(dataframes[TEST], transform=TRANSFORM, selected_features=individual)}

    dataloaders = {TRAIN: DataLoader(datasets[TRAIN], config['model']['train_batch_size'], shuffle=True, generator=rng),
                  VAL: DataLoader(datasets[VAL], config['model']['val_batch_size'], shuffle=False, generator=rng),
                  TEST: DataLoader(datasets[TEST], config['model']['test_batch_size'], shuffle=False, generator=rng)
                  }

    # Train and evaluate the model (example function)
    model = BaseCNNModel().to(config['device'])

    criterion = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=config['model']['learning_rate'], weight_decay=0.0002)

    model, score = train_model(model, criterion, optimizer, dataloaders, num_epochs = 10, display_disabled=display_disabled)

    fitness_score = 1 / (1 + score)


    return fitness_score

def tournament_select(population, fitness_scores):
    """Select one parent using tournament selection"""
    tournament_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    tournament_fitness = [(i, fitness_scores[i]) for i in tournament_indices]
    winner_idx = max(tournament_fitness, key=lambda x: x[1])[0]
    return population[winner_idx]

def genetic_algorithm(dataframes, config):
    population = generate_population()

    for generation in tqdm(range(NUM_GENERATIONS), total=NUM_GENERATIONS, position=0, leave=True):
        fitness_scores = [(individual, fitness(individual, dataframes, config=config)) 
                         for individual in tqdm(population, total=len(population), position=1)]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        fitness_scores_df = pd.DataFrame(fitness_scores, columns=['features', 'fitness_score'])
        fitness_scores_df.to_csv(f'Data/fitness_scores_gen{generation}.csv', index=False)

        print(f'Top 5 fitness scores generation {generation}: \n', fitness_scores_df[:5])

        # Elitism - keep best individuals
        new_population = [ind for ind, _ in fitness_scores[:ELITE_SIZE]]

        # Tournament selection for remaining spots
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_select(population, fitness_scores_df['fitness_score'])
            parent2 = tournament_select(population, fitness_scores_df['fitness_score'])
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

        # Trim to exact population size if needed
        population = new_population[:POPULATION_SIZE]

    return fitness_scores_df
