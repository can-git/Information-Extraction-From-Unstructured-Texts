import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
import Properties as p
from InitializationLayer import InitializationLayer
from tqdm import tqdm
from TextDataset import TextDataset
from Preprocess import Preprocess


def train(model, train_data, val_data):
    train, val = TextDataset(train_data, True), TextDataset(val_data, True)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=p.BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=p.BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=p.LR)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(p.EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            # output = model(input_id)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')


def evaluate(model, test_data):
    test = TextDataset(test_data, False)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            # print(output[0])



            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    # new_df = pd.DataFrame({"img_id": names, "cancer_score": predictions})
    # new_df.to_csv("can_yilmaz_assignment_1.csv", index=False)

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


df_train, df_val, df_test = Preprocess().getItem()

model = InitializationLayer()
train(model, df_train, df_val)

evaluate(model, df_test)

