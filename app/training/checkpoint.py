import torch
from app.config.constant import DEVICE

def saveCheckpoint(model, optimizer, epoch, loss, filename):
    print(f'Saving checkpoint for epoch {epoch}...')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }

    torch.save(checkpoint, filename)
    print('Checkpoint saved succesfully')


def loadCheckpointEpoch(model, optimizer, filename):
    # Load dictionary
    checkpoint = torch.load(filename)

    # Restore parameters + optimizer's state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.to(DEVICE)
    
    print(f"Model loaded from epoch {epoch} with loss: {loss:.4f}")
    
    # Return the restored epoch number so you can resume training from there
    return epoch + 1

def loadModel(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval() 
    model.to(DEVICE)
    
    print("✅ Model loaded successfully and ready for inference.")
    return model