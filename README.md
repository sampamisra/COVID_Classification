# COVID_Classification
Source code
    Dataset
          Raw_Test
          Refined_Test
      Model
            Checkpoint    
      Python_Code
           Test_code_ensemble2.py
          Test_code_ensemble3.py
Before running the program one has to keep all the folders and files (Dataset, Model, and Python_Code) in the same folder
               Dataset:
                 Test Data : (change the dataset if you have new X-ray image to classify)
                            Normal; Pneumonia; Covid
                Model:
                        checkpoint_D1.pt (classify normal vs disease)
                        checkpoint_D2.pt (classify pneumonia vs non-pneumonia)
                        checkpoint_D3.pt ( classify covid or not)
             checkpoint_ensemble<D1D2D3>.pt  (classify covid or not) 
            (Either any 2 or all 3 checkpoints can be combined; for e.g.,  if D1 and D2 is combined the model will be checkpoint_ensemble12.pt   )
        combine 3 checkpoints, it will be checkpoint_ensemble123.pt  )
                    
              Python program:
                Test_code_ensemble2.py
                     This the main program to check our model’s performance for combining 2 checkpoints
1.	Load the checkpoints which models you want to combine and then ensemble them.
For example, if you want to combine D2 and D3 then in Test_code_ensemble2.py
  modelA.load_state_dict(torch.load('checkpoint_D2.pt')) 
modelB.load_state_dict(torch.load('checkpoint_D3.pt'))

model = MyEnsemble(modelB, modelC)

……
model.load_state_dict(torch.load('Model\checkpoint_ensemble23.pt'))


2.	Run the Test_code_ensemble2.py code
                             
