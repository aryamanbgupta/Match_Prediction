# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

# %%
from pathlib import Path
import json

basepath = Path('/Users/aryamangupta/CricML/Match_Prediction/')
data_folder_path = basepath / 'data' / 'ipl_json'


    
                


        

# %%
stadium_to_city = {
    # Delhi
    'Arun Jaitley Stadium': 'Delhi_A',
    'Arun Jaitley Stadium, Delhi': 'Delhi_A',
    'Feroz Shah Kotla': 'Delhi_A',  # This is the old name of Arun Jaitley Stadium
    
    # Mumbai (3 different stadiums)
    'Brabourne Stadium': 'Mumbai_A',
    'Brabourne Stadium, Mumbai': 'Mumbai_A',
    'Dr DY Patil Sports Academy': 'Mumbai_B',
    'Dr DY Patil Sports Academy, Mumbai': 'Mumbai_B',
    'Wankhede Stadium': 'Mumbai_C',
    'Wankhede Stadium, Mumbai': 'Mumbai_C',
    
    # Pune (2 different stadiums)
    'Maharashtra Cricket Association Stadium': 'Pune_A',
    'Maharashtra Cricket Association Stadium, Pune': 'Pune_A',
    'Subrata Roy Sahara Stadium': 'Pune_A',  # Old name of MCA Stadium
    'Nehru Stadium': 'Pune_B',
    
    # Mohali/Chandigarh area (2 different stadiums)
    'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur': 'Mohali_B',
    'Punjab Cricket Association IS Bindra Stadium': 'Mohali_A',
    'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Mohali_A',
    'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Mohali_A',
    'Punjab Cricket Association Stadium, Mohali': 'Mohali_A',
    
    # Abu Dhabi (same stadium, different names)
    'Sheikh Zayed Stadium': 'Abu_Dhabi_A',
    'Zayed Cricket Stadium, Abu Dhabi': 'Abu_Dhabi_A',
    
    # Other Indian Cities (single stadium each)
    'Barabati Stadium': 'Cuttack',
    'Barsapara Cricket Stadium, Guwahati': 'Guwahati',
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow': 'Lucknow',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 'Visakhapatnam',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 'Visakhapatnam',
    'Eden Gardens': 'Kolkata',
    'Eden Gardens, Kolkata': 'Kolkata',
    'Green Park': 'Kanpur',
    'Himachal Pradesh Cricket Association Stadium': 'Dharamsala',
    'Himachal Pradesh Cricket Association Stadium, Dharamsala': 'Dharamsala',
    'Holkar Cricket Stadium': 'Indore',
    'JSCA International Stadium Complex': 'Ranchi',
    'M Chinnaswamy Stadium': 'Bengaluru',
    'M Chinnaswamy Stadium, Bengaluru': 'Bengaluru',
    'M.Chinnaswamy Stadium': 'Bengaluru',
    'MA Chidambaram Stadium': 'Chennai',
    'MA Chidambaram Stadium, Chepauk': 'Chennai',
    'MA Chidambaram Stadium, Chepauk, Chennai': 'Chennai',
    'Narendra Modi Stadium, Ahmedabad': 'Ahmedabad',
    'Sardar Patel Stadium, Motera': 'Ahmedabad',  # Old name of Narendra Modi Stadium
    'Rajiv Gandhi International Stadium': 'Hyderabad',
    'Rajiv Gandhi International Stadium, Uppal': 'Hyderabad',
    'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Hyderabad',
    'Saurashtra Cricket Association Stadium': 'Rajkot',
    'Sawai Mansingh Stadium': 'Jaipur',
    'Sawai Mansingh Stadium, Jaipur': 'Jaipur',
    'Shaheed Veer Narayan Singh International Stadium': 'Raipur',
    'Vidarbha Cricket Association Stadium, Jamtha': 'Nagpur',
    
    # UAE Stadiums (other)
    'Dubai International Cricket Stadium': 'Dubai',
    'Sharjah Cricket Stadium': 'Sharjah',
    
    # South African Stadiums
    'Buffalo Park': 'East_London',
    'De Beers Diamond Oval': 'Kimberley',
    'Kingsmead': 'Durban',
    'New Wanderers Stadium': 'Johannesburg',
    'Newlands': 'Cape_Town',
    'OUTsurance Oval': 'Bloemfontein',
    "St George's Park": 'Port_Elizabeth',
    'SuperSport Park': 'Centurion'
}

# %%
dataset = []


for json_path in sorted(data_folder_path.glob('*.json')):
    with json_path.open() as f:
        json_data = f.read()
        match_data = json.loads(json_data)
        # print (data['info'])
        year = match_data["info"]["dates"][0].split('-')[0]
        # print(year)
        stadium = stadium_to_city[match_data["info"]["venue"]]

        for inning in match_data['innings']:
            inning_dataset = []
            team_runs = 0
            for over in inning['overs']:
                for delivery in over['deliveries']:
                    delivery_data = []
                    #Delivery outcome
                    if 'wickets' in delivery:
                        delivery_data.append('W')
                    else:
                        #Added cases to remove uncommon classes from the dataset, can be added back later
                        if delivery['runs']['total'] == 3:
                            delivery_data.append('2')
                        elif delivery['runs']['total'] == 5:
                            delivery_data.append('4')
                        elif delivery['runs']['total'] == 7:
                            delivery_data.append('6')
                        else:
                            delivery_data.append(str(delivery['runs']['total']))

                    delivery_data.append(stadium)

                    batter = delivery["batter"]
                    batter_id = match_data["info"]["registry"]["people"][batter]
                    delivery_data.append(batter_id)

                    bowler = delivery["bowler"]
                    bowler_id = match_data["info"]["registry"]["people"][bowler]
                    delivery_data.append(bowler_id)

                    non_striker = delivery["non_striker"]
                    non_striker_id = match_data["info"]["registry"]["people"][non_striker]
                    delivery_data.append(non_striker_id)

                    team_runs += delivery['runs']['total']
                    delivery_data.append(team_runs)
                    
                    inning_dataset.append(delivery_data)
                    
            dataset.append((inning_dataset,year))

# %%
print(len(dataset))
print((dataset[0][0]))

# %%
# print(dataset[0][11])
all_runs = [balls[0] for inning,year in dataset for balls in inning]
runs = sorted(list(set(all_runs)))
print(runs)
stoi = {s:i+1 for i,s in enumerate(runs)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)

n_outcomes = len(runs)+1
print(n_outcomes)


# %%
# Count frequencies using a Counter
from collections import Counter
outcome_counts = Counter(all_runs)

# Print frequencies in a readable format
print("\nOutcome frequencies:")
for outcome in sorted(outcome_counts.keys()):
    print(f"{outcome}: {outcome_counts[outcome]}")

# You can also get total count to verify
total_balls = sum(outcome_counts.values())
print(f"\nTotal number of balls: {total_balls}")

n_outcomes = len(runs)+1
print(f"\nNumber of unique outcomes: {n_outcomes}")

# %%
#stadium embeddings
# TO DO:add one stadium type for unrecognised stadium
all_stadiums = [balls[1] for inning,year in dataset for balls in inning]
stadiums = sorted(list(set(all_stadiums)))
print(stadiums)
stadiumtoi = {s:i for i,s in enumerate(stadiums)}
#stadiumtoi['.'] = 0
itostadium = {i:s for s,i in stadiumtoi.items()}
print(itostadium)

n_stadiums = len(stadiums)
print(n_stadiums)

# %%
#batter embeddings
# TO DO:add one batter type for unrecognised batter
all_batters = [balls[2] for inning,year in dataset for balls in inning]
non_strikers = [balls[4] for inning,year in dataset for balls in inning]
batters = sorted(list(set(all_batters + non_strikers)))
print(batters)
battertoi = {s:i+1 for i,s in enumerate(batters)}
battertoi['.'] = 0
itobatter = {i:s for s,i in battertoi.items()}
print(itobatter)

n_batters = len(batters)
print(n_batters)

# %%
#bowler embeddings
# TO DO:add one bowler type for unrecognised bowler
all_bowlers = [balls[3] for inning,year in dataset for balls in inning]
bowlers = sorted(list(set(all_bowlers)))
print(bowlers)
bowlertoi = {s:i+1 for i,s in enumerate(bowlers)}
bowlertoi['.'] = 0
itobowler = {i:s for s,i in bowlertoi.items()}
print(itobowler)

n_bowlers = len(bowlers)
print(n_bowlers)

# %%
years = [year for match,year in dataset for run in match]
year_count = {}
for yearn in years:
    if yearn not in year_count:
        year_count[yearn]=1
    else:
        year_count[yearn]+=1

total_deliveries = sum(year_count.values())
print(total_deliveries)
year_percentages = {year: (count/total_deliveries)*100 for year,count in year_count.items()}



# %%
sorted_years = sorted(year_count.keys())
counts = [year_count[year] for year in sorted_years]
percentages = [year_percentages[year] for year in sorted_years]

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.tight_layout(pad=3.0)

# Plot absolute counts
ax1.bar(sorted_years, counts, color='skyblue')
ax1.set_title('Absolute Count by Year')
ax1.set_xlabel('Year')
ax1.set_ylabel('Count')
ax1.grid(True, alpha=0.3)

# Plot percentages
ax2.bar(sorted_years, percentages, color='lightgreen')
ax2.set_title('Percentage Distribution by Year')
ax2.set_xlabel('Year')
ax2.set_ylabel('Percentage (%)')
ax2.grid(True, alpha=0.3)

# Add value labels on top of each bar
for ax in [ax1, ax2]:
    for i, v in enumerate(counts if ax == ax1 else percentages):
        ax.text(sorted_years[i], v, f'{v:.1f}', 
                ha='center', va='bottom')

plt.show()

# %%
block_size = 6
def create_xy_pairs(data, block_size, stoi, stadiumtoi):
    X_outcome, X_stadium, X_batter, X_bowler, X_non_striker, Y = [], [], [], [], [], []  # separate lists for outcomes and stadiums
    for match in data:
        ix_stadium = stadiumtoi[match[0][0][1]]
        context_outcome = [0] * block_size
        context_stadium = [ix_stadium] * block_size  # add stadium context
        context_batter = [0]*block_size
        context_bowler = [0]*block_size
        context_non_striker = [0]*block_size
        for ball in match[0]:
            outcome = ball[0]
            stadium = ball[1]
            batter = ball[2]
            bowler = ball[3]
            non_striker = ball[4]
            
            ix = stoi[outcome]
            ix_stadium = stadiumtoi[stadium]
            ix_batter = battertoi[batter]
            ix_bowler = bowlertoi[bowler]
            ix_non_striker = battertoi[non_striker]

            X_outcome.append(context_outcome)
            X_stadium.append(context_stadium)  # append stadium context
            X_batter.append(context_batter)
            X_bowler.append(context_bowler)
            X_non_striker.append(context_non_striker)
            Y.append(ix)
            
            context_outcome = context_outcome[1:] + [ix]
            context_batter = context_batter[1:] + [ix_batter]
            context_bowler = context_bowler[1:] + [ix_bowler]
            context_non_striker = context_non_striker[1:] + [ix_non_striker]
            # context_stadium = context_stadium[1:] + [ix_stadium]  # update stadium context
            
    return torch.tensor(X_outcome), torch.tensor(X_stadium), torch.tensor(X_batter), torch.tensor(X_bowler), torch.tensor(X_non_striker), torch.tensor(Y)

# Split data based on years
train_data = [inning for inning in dataset if int(inning[1]) <= 2020]
val_data = [inning for inning in dataset if int(inning[1]) in [2021, 2022]]
test_data = [inning for inning in dataset if int(inning[1]) >= 2023]

# Create X,Y pairs for each split
X_train_outcome, X_train_stadium, X_train_batter, X_train_bowler, X_train_non_striker, y_train = create_xy_pairs(train_data, block_size, stoi, stadiumtoi)
X_val_outcome, X_val_stadium, X_val_batter, X_val_bowler, X_val_non_striker, y_val = create_xy_pairs(val_data, block_size, stoi, stadiumtoi)
X_test_outcome, X_test_stadium, X_test_batter, X_test_bowler, X_test_non_striker, y_test = create_xy_pairs(test_data, block_size, stoi, stadiumtoi)

# %%
X_train_stadium.shape, X_train_stadium.dtype, y_train.shape, y_train.dtype
# X_val.shape[0] + X_train.shape[0] + X_test.shape[0]

# %%
n_embed_outcome = 2
n_embed_stadium = 2
n_embed_batter = 3
n_embed_bolwer = 3


# %%
g = torch.Generator().manual_seed(428)
C_outcome = torch.randn ((n_outcomes,n_embed_outcome), generator=g)
C_stadium = torch.randn ((n_stadiums,n_embed_stadium), generator=g)
C_batter = torch.randn ((n_batters,n_embed_batter), generator=g)
C_bowler = torch.randn((n_bowlers, n_embed_bolwer), generator=g)
W1 = torch.randn((block_size*(n_embed_outcome + n_embed_stadium + n_embed_batter + n_embed_bolwer + n_embed_batter),300), generator=g)
b1 = torch.randn(300, generator=g)
W2 = torch.randn((300,n_outcomes),generator=g)*0.1
b2 = torch.randn(n_outcomes, generator=g)*0
parameters = [C_outcome, C_stadium, C_batter, W1, b1, W2, b2]

# %%
sum(p.nelement() for p in parameters)

# %%
for p in parameters:
    p.requires_grad = True

# %%
lre = torch.linspace(-2,0,1000)
lrs = 10**lre

# %%
#Add learning rate decay

lri = []
lossi = []

for i in range (1000):

    #minibatch construction
    ix = torch.randint(0,X_train_outcome.shape[0],(32,))
    #forward pass
    emb_outcome = C_outcome[X_train_outcome[ix]]
    emb_stadium = C_stadium[X_train_stadium[ix]]
    emb_batter = C_batter[X_train_batter[ix]]
    emb_bowler = C_bowler[X_train_bowler[ix]]
    emb_non_striker = C_batter[X_train_non_striker[ix]]
    emb = torch.cat([emb_outcome, emb_stadium, emb_batter, emb_bowler, emb_non_striker], dim=-1) 
    h = torch.tanh(emb.view(-1,block_size*(n_embed_outcome + n_embed_stadium + n_embed_batter + n_embed_bolwer + n_embed_batter)) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y_train[ix])
    # print(loss.item())
    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    #update
    # lr = lrs[i]
    lr = 10**-1.5
    for p in parameters:
        p.data += -lr * p.grad

    #track stats
    # lri.append(lre[i])
    lossi.append(loss.log10().item())

# %%
plt.hist(h.view(-1).tolist(),50)

# %%
with torch.no_grad():
    probs = torch.softmax(logits[0], dim=0)
    print(probs)#torch.log(probs).numpy())

# %%
plt.plot(lossi)

# %%
emb_outcome = C_outcome[X_train_outcome]
emb_stadium = C_stadium[X_train_stadium]
emb_batter = C_batter[X_train_batter]
emb_bowler = C_bowler[X_train_bowler]
emb_non_striker = C_batter[X_train_non_striker]
emb = torch.cat([emb_outcome, emb_stadium, emb_batter, emb_bowler, emb_non_striker], dim=-1) 
h = torch.tanh(emb.view(-1,block_size*(n_embed_outcome + n_embed_stadium + n_embed_batter + n_embed_bolwer + n_embed_batter)) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, y_train)
loss

# %%
emb_outcome = C_outcome[X_val_outcome]
emb_stadium = C_stadium[X_val_stadium]
emb_batter = C_batter[X_val_batter]
emb_bowler = C_bowler[X_val_bowler]
emb_non_striker = C_batter[X_val_non_striker]
emb = torch.cat([emb_outcome, emb_stadium, emb_batter, emb_bowler, emb_non_striker], dim=-1) 
h = torch.tanh(emb.view(-1,block_size*(n_embed_outcome + n_embed_stadium + n_embed_batter + n_embed_bolwer + n_embed_batter)) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, y_val)
loss

# %%
plt.figure(figsize=(8,8))
plt.scatter (C_stadium[:,0].data, C_stadium[:,1].data, s=200)
for i in range (36):
    plt.text(C_stadium[i,0].item(), C_stadium[i,1].item(), itostadium[i], ha="center",va = "center" , color = "black")
plt.grid('minor')

# %%



