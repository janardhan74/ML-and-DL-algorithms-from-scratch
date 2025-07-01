import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import ALL_LETTERS , N_LETTERS
from utils import load_data , letter_to_tensor , line_to_tensor , random_training_example

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN,self).__init__()
        
        self.hidden_size = hidden_size
        
        self.i20 = nn.Linear(input_size + hidden_size , output_size)
        self.i2h = nn.Linear(input_size + hidden_size , hidden_size)
        
        self.softmax = nn.LogSoftmax(dim=1) # the input size for softmax is [batch,classes] , we want to apply the softmax in the directions of classes to we dim = 1
        # along rows -> dim = 0
        # along cols -> dim = 1
        
        
    def forward(self,input_tensor,hidden_tensor):
        combined = torch.cat((input_tensor,hidden_tensor),dim=1) # [input_tensor | hidden_tensor] -> [(1,input_size),(1,hidden_size)] -> [1,input_size+hidden_size] -> so dim = 1
        
        hidden = self.i2h(combined)
        output = self.i20(combined)
        output = self.softmax(output)
        
        return output , hidden
    
    def init_hidden(self):
        return torch.zeros(size=[1,self.hidden_size])
    

category_lines , all_categories = load_data()
n_categories = len(all_categories)
print(f"N categories : {n_categories}")

hidden_size = 128 # hyper parameter
rnn = RNN(input_size=N_LETTERS,
          hidden_size=hidden_size,
          output_size=n_categories)

# ipt = letter_to_tensor("J")

# ht = rnn.init_hidden()
# ot,ht = rnn(ipt,ht)
# print(ot.shape)
# print(ht.shape)

def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]

learning_rate = 0.005 # hyper parameter
criterion = nn.NLLLoss() # Negative Log likely hood (remember cross etropy formula and log-likelihood formula) # since we used LogSoftmax the probs are represented in log form 
optimizer = torch.optim.SGD(rnn.parameters(),lr=learning_rate)

def train(line_tensor,category_tensor):
    hidden = rnn.init_hidden()
    
    for i in range(line_tensor.shape[0]): # iterate over number of letters
        output,hidden = rnn(line_tensor[i],hidden)
        
    optimizer.zero_grad()
    loss = criterion(output,category_tensor)
    loss.backward()
    optimizer.step()
    
    return output , loss.item()

current_loss = 0
all_lossess = []
plot_steps , print_steps = 1000,5000
n_iters = 100000

for i in range(n_iters):
    category , line , category_tensor , line_tensor = random_training_example(category_lines , all_categories)
    
    output , loss = train(line_tensor , category_tensor)
    
    if (i%plot_steps) == 0:
        all_lossess.append(loss)
        current_loss = 0
        
    if (i%print_steps) == 0:
        print(f"{i/n_iters*100}% completed \n Loss : {loss:.4f} \n Preicted : {category_from_output(output)} True : {category}")
        
plt.figure()
plt.plot(all_lossess)

while True:
    line = input("Enter the name : ")
    if line == "quit":
        break
    line_tensor = line_to_tensor(line)
    hidden = rnn.init_hidden()
    
    for i in range(line_tensor.shape[0]):
        output,hidden = rnn(line_tensor[i],hidden)
        
    print(f"Predicted : {category_from_output(output)}")
