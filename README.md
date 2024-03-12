# NQAS

## For Training

```
python train.py --learning_rate 3e-4 --num_train_epochs 5 --per_device_train_batch_size 8 \
    --model_name_or_path distilbert-base-uncased --dataset_name squad  --output_dir < enter output directory for model saving > \
     --do_train --do_eval 
```

## For Inference

Model can be found in [google drive](https://drive.google.com/drive/folders/1cXHNbh_ZTOPJTheSmrM6WNAc1RMxbbqw?usp=sharing)
edit model path in test.py
edit question and context variables in test.py

```
python test.py
```

## Real-Time Conversation Agent

Google CCAI Agent Interface for conversation streaming with an agent in real-time. You can find the agent as ``` agent.json ``` with some examples.
