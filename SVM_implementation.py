# loss function being used here is l = 2 * lambda * ||w||^2 + \sum max(0, 1 - y_i(wx + b)))
# I'm using gradient descent algorithm


# Data generation code (was generated using ChatGPT),
# this code block is being used by the entire team.

# Parameters for class 0
mean_class0 = [2, 2]  # Mean for class 0
cov_class0 = [[0.5, 0], [0, 0.5]]  # Reduced covariance for clear separation
 
# Parameters for class 1
mean_class1 = [6, 6]  # Mean for class 1, far apart from class 0
cov_class1 = [[0.5, 0], [0, 0.5]]  # Same covariance as class 0
 
# Generate data
np.random.seed(42)  # For reproducibility
class0_data = np.random.multivariate_normal(mean_class0, cov_class0, 100)
class1_data = np.random.multivariate_normal(mean_class1, cov_class1, 100)
 
# Create labels
labels_class0 = np.zeros(100)
labels_class1 = np.ones(100)
 
# Combine data into a pandas DataFrame
data_class0 = pd.DataFrame(class0_data, columns=['Feature1', 'Feature2'])
data_class0['Label'] = labels_class0
 
data_class1 = pd.DataFrame(class1_data, columns=['Feature1', 'Feature2'])
data_class1['Label'] = labels_class1
 
# Combine both classes into one DataFrame
data = pd.concat([data_class0, data_class1], ignore_index=True)



# loss function being used here is l = 2 * lambda * ||w||^2 + \sum max(0, 1 - y_i(wx + b)))
# I'm using gradient descent algorithm

# loss function being used here is l = 2 * lambda * ||w||^2 + \sum max(0, 1 - y_i(wx + b)))
# I'm using gradient descent algorithm

class SVM:
    def __init__(self, step_size=0.001,
                 lambda_param=1,
                 n_iters=100,
                 batch_size = 32):
        self.ss = step_size
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        # self.w = None
        # self.b = None
        self.w = torch.randn(2)
        self.b = torch.randn(1)
        self.batch_size = batch_size

    def fit(self, x, y):
        n_samples, n_features = x.shape

        # I'm doing the tanspose to make them look like vectors.
        # personal preference.
        self.w = torch.randn(n_features)
        self.b = torch.randn(1)
        y = np.where(y == 0, -1, 1)
        total_w_grad = 0
        total_b_grad = 0


        x_torch = torch.from_numpy(x)
        y_torch = torch.from_numpy(y).unsqueeze(dim = 1)

        # Creating a torch dataloader for mini-batch gradient descent.
        dataset_1 = torch.utils.data.TensorDataset(x_torch, y_torch)
        
        dataloader = DataLoader(dataset = dataset_1,
                                batch_size=self.batch_size,
                                shuffle=True)
        # l1 = []
        # l2 = []
        # for batch, (x_batch, y_batch) in enumerate(dataset_1):
        #     l1.append(x_batch[0])
        #     l2.append(x_batch[1])
        # plt.scatter(l1, l2)
        # return
        for i in range(self.n_iters):

            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                w_grad_correct = self.lambda_param * self.w
                w_repeated = self.w.repeat(self.batch_size, 1)
                b_repeated = self.b.repeat(self.batch_size, 1)
                # print(x_batch.shape)
                # print(y_batch.shape)
                # print(w_repeated.shape)
                # print(b_repeated.shape)
                # print(x_batch)
                # print(y_batch)
                # print(w_repeated)
                # print(b_repeated)
                # print("our stuff is below this right now.")
                # print(torch.sum(w_repeated * x_batch, dim=1)
                f = torch.sum(w_repeated * x_batch, dim=1, keepdim=True) + b_repeated
                # print(f)
                reshaped = y_batch.reshape(-1, 1)
                y_f =  reshaped * f

                for i in range(len(y_f)):
                    if y_f[i][0] < 1:
                        total_w_grad += (w_grad_correct - reshaped[i] * x_batch[i])
                        #print(total_w_grad)
                        total_b_grad += reshaped[i]
                    else:
                        total_w_grad += w_grad_correct
                        #print(total_w_grad)
                        total_b_grad += 0

                mini_batch_grad_w = total_w_grad / self.batch_size
                mini_batch_grad_b = total_b_grad / self.batch_size
                self.w = self.w + self.ss * mini_batch_grad_w
                self.b = self.b + self.ss * mini_batch_grad_b
                total_w_grad = 0
                total_b_grad = 0





    def predict(self, X):
        pass