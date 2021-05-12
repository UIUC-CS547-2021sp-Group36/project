

#create trainer SGD, ATOM, etc. setup traingin parameters
#create loss function
loss_function = LossFunction(g)
#load dataset, index, create dataloader from dataset and index.

for epoch in range(NUM_EPOCHS):
    
    #clear gradient
    #whatever epoch setup.
    #epoch loss = 0
    
    #batch loop
    for (Qs,Ps,Ns),labels in dataloader:
        
        #Qs is a tensor containing images. So if images are 64x64 in 3 colors,
        #the size of Q will be (BATCH_SIZE, 3, 64, 64)
        
        
        #this is a single batch.
        #model.clear_gradient()
        #trainer
        
        Q_embedding_vectors = model(Qs)
        P_embedding_vectors = model(Ps)
        N_embedding_vectors = model(Ns)
        
        #Embedding vectors will individually be of shape (D_EMBEDDING),
        #but the tensor named Q_embedding_vectors will be a tensor of shape
        # (BATCH_SIZE, D_EMBEDDING)
        
        batch_loss = loss_function(Q_embedding_vectors,P_embedding_vectors,N_embedding_vectors)
        batch_loss.backward()
        
        trainer.step()
    
    if epoch % 10:
        #test validation loss
        #do some output
        #whatever
