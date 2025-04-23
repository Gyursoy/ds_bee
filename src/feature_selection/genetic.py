import random
from collections import defaultdict
import torch
from tqdm.auto import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import torch.nn as nn
from torchvision import transforms
import logging
from datetime import datetime
import os

from ..training.train import train_model, get_scheduler
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

ELITE_SIZE = int(POPULATION_SIZE * 0.05)  # Keep top 10% of individuals
TOURNAMENT_SIZE = 5  # Number of individuals in each tournament

log_dir = 'Data/ga_logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/ga_run_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
        logging.StreamHandler()
    ]
)

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
            
            if available_features:
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

        # Remove duplicates
        combined_features = list(dict.fromkeys(combined_features))
        
        # Randomly select features ensuring no duplicates
        for i in range(TARGET_FEATURE_COUNT):

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

    # Train and evaluate the model
    model = BaseCNNModel().to(config['device'])

    criterion = nn.MSELoss()

    # optimizer = Adam(model.parameters(), lr=config['model']['learning_rate'], weight_decay=0.0002)
    optimizer = AdamW(model.parameters(), lr=config['model']['learning_rate'], weight_decay=config['model']['weight_decay'])
    
    scheduler = get_scheduler(
    optimizer,
    scheduler_type='one_cycle',
    max_lr=0.001,  # adjust based on your needs
    steps_per_epoch=len(dataloaders[TRAIN])
    )

    epochs = config['model']['num_epochs']

    model, score = train_model(model, criterion, optimizer, dataloaders, num_epochs=epochs, display_disabled=display_disabled, scheduler=scheduler)

    fitness_score = 1 / (1 + score)

    return fitness_score

def tournament_select(population, fitness_scores):
    """Select one parent using tournament selection"""
    tournament_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    tournament_fitness = [(i, fitness_scores[i]) for i in tournament_indices]
    winner_idx = max(tournament_fitness, key=lambda x: x[1])[0]
    return population[winner_idx]

def string_to_features_dict(features_str):
    # Remove defaultdict wrapper and convert string to dict
    features_str = features_str.replace('defaultdict(<class \'list\'>, ', '').rstrip(')')
    # Convert string to dictionary
    features_dict = eval(features_str)
    # Convert all feature indices to int
    return {k: [int(i) for i in v] for k, v in features_dict.items()}

def genetic_algorithm(dataframes, config, continue_from=None):
    global NUM_GENERATIONS
    
    if isinstance(continue_from, str): 
        if not os.path.exists(continue_from):
            raise FileNotFoundError(f"File {continue_from} does not exist")
        logging.info(f"Continuing from generation {continue_from}")
        population = pd.read_csv(continue_from)['features'].apply(string_to_features_dict).tolist()
        best_fitness = pd.read_csv(continue_from)['fitness_score'].max()
        start_gen = int(continue_from.split('_')[-1].split('.')[0][-1:])
        NUM_GENERATIONS = NUM_GENERATIONS - start_gen
    else:
        logging.info(f"Starting GA with population size: {POPULATION_SIZE}, generations: {NUM_GENERATIONS}")
        logging.info(f"Elite size: {ELITE_SIZE}, Tournament size: {TOURNAMENT_SIZE}")
        population = generate_population()
        best_fitness = float('-inf')
    
    generations_without_improvement = 0



    for generation in tqdm(range(NUM_GENERATIONS), total=NUM_GENERATIONS, position=0, leave=True):
        logging.info(f"\nGeneration {generation} started")
        
        fitness_scores = [(individual, fitness(individual, dataframes, config=config)) 
                         for individual in tqdm(population, total=len(population), position=1)]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # Log generation statistics
        current_best_fitness = fitness_scores[0][1]
        avg_fitness = sum(score for _, score in fitness_scores) / len(fitness_scores)
        logging.info(f"Generation {generation} stats:")
        logging.info(f"Best fitness: {current_best_fitness:.6f}")
        logging.info(f"Average fitness: {avg_fitness:.6f}")

        # Track improvement
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            generations_without_improvement = 0
            logging.info(f"New best fitness found: {best_fitness:.6f}")
        else:
            generations_without_improvement += 1
            logging.info(f"Generations without improvement: {generations_without_improvement}")

        fitness_scores_df = pd.DataFrame(fitness_scores, columns=['features', 'fitness_score'])
        fitness_scores_df.to_csv(f'Data/ga_results/exp1/fitness_scores_gen{generation}.csv', index=False)

        print(f'Top 5 fitness scores generation {generation}: \n', fitness_scores_df[:5])

        # Elitism - keep best individuals
        new_population = [ind for ind, _ in fitness_scores[:ELITE_SIZE]]

        # Tournament selection for remaining spots
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_select(population, fitness_scores_df['fitness_score'])
            parent2 = tournament_select(population, fitness_scores_df['fitness_score'])
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population[:POPULATION_SIZE]

        if generations_without_improvement >= 8:
            logging.info("Stopping early due to no improvement")
            break

    logging.info("Genetic Algorithm completed")
    logging.info(f"Best fitness achieved: {best_fitness:.6f}")
    return fitness_scores_df
