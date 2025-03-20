# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="aab1aa45"
# #  Visualizing Adoption and Return Trends in Sonoma Animal Data
# 03/15/2024

# %% [markdown] id="8d17ef9f"
# #### Rafael L.S. Reis, Dalia Cabrera Hurtado, Gabe Myers

# %% [markdown] id="dd9d8a1b"
# ## Introduction
# The Sonoma Animal Shelter dataset, provided by the County of Sonoma Department of Health Services, comprises about 30,000 records detailing various attributes of animals admitted to the shelter. Each record includes species, breed, age, sex, and color, along with intake and outcome types to track the animals journey in the shelter. With over 4,000 animals entering the shelter annually, the dataset offers insights into adoption trends, shelter capacity, and animal welfare efforts. It gets updated regularly with the most recent being March 18, 2025.
#
# This analysis seeks to answer two key questions: first, how does the number of days an animal spends in the shelter differ between those that are adopted and those that are returned to their owners; and second, is there an association between an animal's primary coat color—extracted from compound color entries—and its outcome or duration of shelter stay.
#
# note: This is tentative and we will likely hone down our scope of questionings(likely to colors)
#
# data downloaded from:
# https://raw.githubusercontent.com/grbruns/cst383/master/sonoma-shelter-15-october-2024.csv

# %% id="fdf3ed81"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime # for age stuff

# %% [markdown] id="f03dd8de"
# ## Initial Data Exploration

# %% id="8694f864"
df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/sonoma-shelter-17-march-2025.csv')

# %% [markdown] id="d1283d9a"
# ## Data preprocessing

# %% id="85iYy59zSfzm" outputId="b0cffc8d-33bc-4a95-a8b7-02511d982bfc" colab={"base_uri": "https://localhost:8080/"}
df.info()

# %% id="lNr84NVtSjtG" outputId="f2363223-5fcd-4c64-8939-4eb98d3cbf3f" colab={"base_uri": "https://localhost:8080/", "height": 714}
df.sample(5)


# %% [markdown] id="K1CCaeqrSsuf"
# Among dogs returned to their owners, which coat colors are most frequently associated with being lost or having escaped? To try explore this, we can create a simplified color category column to analyze color hues effectively.

# %% [markdown] id="5clMunrXW8SN"
# Below are the functions to help create a new column to get the color shade of the animals

# %% [markdown]
# Function to clean and normalize breeds

# %% [markdown]
# ### Functions and expressions to pre-process data

# %%
# Function to extract primary breed
def get_primary_breed(breed_string):
    if pd.isna(breed_string):
        return "Unknown"
    
    breed_string = str(breed_string).strip()
    
    # For breeds with "MIX" or similar suffix
    if 'MIX' in breed_string: # If mix is the primary 'breed'
        return breed_string.replace('MIX', '').strip()

    if '/' in breed_string:
        return breed_string.split('/')[0].strip() 
        
    return breed_string

# Function to extract primary breed and treat any compund mix as just 'MIX'
def get_primary_breed_mix(breed_string):
    if pd.isna(breed_string):
        return "Unknown"

    if 'mix' in breed_string.lower():
        return 'MIX' # Treat mixes and compund mixes as just 'MIX'
    
    breed_string = str(breed_string).strip()

    if '/' in breed_string:
        return breed_string.split('/')[0].strip()  # else return the primary 'breed'
    
    return breed_string

# Function to calculate age in years using 'MM/DD/YYYY'
def calculate_age(dob_str):
    try:
        if pd.isna(dob_str) or dob_str == "":
            return np.nan
        dob = datetime.strptime(dob_str, '%m/%d/%Y')
        age_in_years = (current_date - dob).days / 365.25
        return age_in_years
    except:
        return np.nan



# %% [markdown]
# ## New Filtered columns

# %%
       
# ------- Columns/Filters for Breed and Days in Shelter Analysis ----------

# Create a new column for primary breed and primary breed with mix generalization
df['PrimaryBreed'] = df['Breed'].apply(get_primary_breed)
df['PrimaryBreedMix'] = df['Breed'].apply(get_primary_breed_mix)

# Separate dogs and cats
dog_df = df[df['Type'] == 'DOG']
cat_df = df[df['Type'] == 'CAT']

# Count the most common dog breeds
dog_breed_counts = dog_df['PrimaryBreed'].value_counts()
most_common_dog_breeds = dog_breed_counts.head(10).index.tolist()

dog_breed_counts_mix = dog_df['PrimaryBreedMix'].value_counts()
most_common_dog_breeds_mix = dog_breed_counts_mix.head(10).index.tolist()


# Get average days for most common dog breeds
dog_breed_days = dog_df[dog_df['PrimaryBreed'].isin(most_common_dog_breeds)]
dog_avg_days = dog_breed_days.groupby('PrimaryBreed')['Days in Shelter'].mean().reindex(most_common_dog_breeds)

dog_breed_days_mix = dog_df[dog_df['PrimaryBreedMix'].isin(most_common_dog_breeds_mix)]
dog_avg_days_mix = dog_breed_days_mix.groupby('PrimaryBreedMix')['Days in Shelter'].mean().reindex(most_common_dog_breeds_mix)

# Count the most common cat breeds
cat_breed_counts = cat_df['PrimaryBreed'].value_counts()
most_common_cat_breeds = cat_breed_counts.head(10).index.tolist()

# Get average days for most common cat breeds
cat_breed_days = cat_df[cat_df['PrimaryBreed'].isin(most_common_cat_breeds)]
cat_avg_days = cat_breed_days.groupby('PrimaryBreed')['Days in Shelter'].mean().reindex(most_common_cat_breeds)


# %%
# ------- Columns/Filters for Age and Days in Shelter Analysis ----------

# Calculate age and filter to ge tonly VALID records for dogs and cats
df['Age'] = df['Date Of Birth'].apply(calculate_age)
animals_df = df.dropna(subset=['Age', 'Days in Shelter'])
animals_df = animals_df[animals_df['Type'].isin(['DOG', 'CAT'])]


# %% [markdown]
# ### Effects of Breed on Days in Shelter for Dog and Cats

# %%
# Plot 1.1: Most Common Dog breeds by average days in shelter
sorted_dog_avg_days = dog_avg_days.sort_values(ascending=False)

# plt.figure(figsize=(12, 6))
plt.bar(dog_avg_days.index, sorted_dog_avg_days.values, width=0.6)
plt.title('Average Days in Shelter for Most Common Dog Breeds')
plt.xlabel('breed')
plt.xticks(rotation=45) 
plt.ylabel('average days')
plt.tight_layout()
plt.savefig('common_dog_breeds_avg_stay.png')
plt.show()


# %%
# Plot 1.2: Most Common Dog breeds by average days in shelter, generalizing mixed breeds
sorted_dog_avg_days_mix = dog_avg_days_mix.sort_values(ascending=False)

plt.bar(dog_avg_days_mix.index, sorted_dog_avg_days_mix.values, width=0.6)
plt.title('Average Days in Shelter for Most Common Dog Breeds')
plt.xlabel('breed')
plt.xticks(rotation=45) 
plt.ylabel('average days')
plt.tight_layout()
plt.savefig('common_dog_breeds_avg_stay.png')
plt.show()

# %%
# Plot 2: Most Common Cat breeds by average days in shelter
sorted_cat_avg_days = cat_avg_days.sort_values(ascending=False)

plt.bar(cat_avg_days.index, sorted_cat_avg_days.values)
plt.title('Average Days in Shelter for Most Common Cat Breeds')
plt.xlabel('breed')
plt.xticks(rotation=45, ha='right') 
plt.ylabel('average days')

# adding count labels -  I think it is important to show this in the cat scenario since there's some low 'n' values
for i, breed in enumerate(sorted_cat_avg_days.index):
    plt.text(i, sorted_cat_avg_days[breed] + 0.5, f"n={cat_breed_counts[breed]}", ha='center')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Effects of Age and Days in Shelter

# %%

# %% [markdown] id="5a3a9d08"
# ## Data exploration and visualization

# %% [markdown] id="oswY20dtVAhI"
# Data above looks good(at least we can see the relative primary colors), but doesn't give us the full picture, maybe there's just more black dogs. Let's explore some more. Out of all dogs of a given shade, what's the proportion successfully returned to their owner?

# %% [markdown]
# ## Size

# %% [markdown]
#  - #Size (should we spearate species?) separate by age(isPuppy, isKitten)? GABE
#     - columns needed: "Days in Shelter", "Size" - kitten/puppy or not?
#     - Violin or figure it out
#     - might have to create new columns - isKitten/isPuppy

# %%
# create a new column to tell if an animal is a puppy or kitten
df["is_puppy_kitten"] = (df["Size"] == "KITTEN") | (df["Size"] == "PUPPY")
# create new column for log and use log1p to handle zeros
df["Days in Shelter_log"] = np.log1p(df["Days in Shelter"]) 
sns.violinplot(x="is_puppy_kitten", y="Days in Shelter_log", data=df)
plt.title("Days in Shelter for Puppies and Kittens (Log-Transformed)")
plt.xlabel("Is Puppy or Kitten")
plt.ylabel("Days in shelter (log scale)")
plt.show();

# %% [markdown]
# These two distributions look pretty simialar execept there is a spike close to 0 for non puppies and kittens. I think this is from peoples animals getting picked up off the street and returned to the owner.

# %% [markdown]
# ## Outcome

# %% [markdown]
# - #Outcome - "how much does each outcome stay?" --> GABE
#     - columns needed: "Outcome Type", "Length of Stay"
#     - Barplot
#     - seems like clean categories 

# %%
df_group = df.groupby("Outcome Type")["Days in Shelter"].mean()
sns.barplot(x="Outcome Type", y="Days in Shelter", df=df_group)
plt.xlabel("Outcome Type")
plt.ylabel("Average Days in Shelter")
plt.title("Average Days in Shelter by Outcome Type")
plt.xticks(rotation=45)
plt.show()

# %%
df.groupby("Outcome Type")["Days in Shelter"].mean().sort_values(ascending=False).plot.bar()
plt.xlabel("Outcome Type")
plt.ylabel("Average Days in Shelter")
plt.title("Average Days in Shelter by Outcome Type")
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# As we can see adopted animals stay the longest with an average of 40 days. This suggests that adoption is a proccess and takes time. I would also assume that the shelter staff are trying to find the best match possible for each dog so some people might get turned away.
# Another interesting thing to note is how quickly animals get returned to their owner on average its about 4 days.

# %% [markdown] id="05ea9473"
# ## Conclusions

# %% [markdown] id="gczUjeHafqXM"
# For now we still need to explore more and improve the notebook. As it stands it's pretty messy but we just wanted to explore as much as we could first and see if we found anything of interest or significance rather than caring too much about form. As we hone down on our areas of interest we will make the data look better and have better descriptions and organization. Lastly exploring the effects of color might be more interesting(given our exploration) so we might pivot to focus more on that.

# %%

# %%

# %%
