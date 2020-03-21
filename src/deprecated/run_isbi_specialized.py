import run_isbi
import predict
import detect_isbi
import data_and_sampling

def cele01_01():
  config = run_isbi.celegans_isbi(train_set='01',pred='01',time='all',tid=2,kernxy=5,kernz=3,)
  config.trainer.sampler = 'flat'
  print(f"Training...")
  T = detect_isbi.init(config.trainer)
  detect_isbi.train(T,config.trainer)
  
  for pset in ['01','02']:
    config = run_isbi.celegans_isbi(train_set='01',pred=pset,time='all',tid=2,kernxy=5,kernz=3,)
    
    print(f"Predicting on {pset}")
    predict.isbi_predict(config.predictor)
    
    print(f"Evaluating on {pset}")
    predict.total_matches(config.evaluator)
    predict.rasterize_isbi_detections(config.evaluator)
    predict.evaluate_isbi_DET(config.evaluator)


def cele01_01():
  config = run_isbi.celegans_isbi(train_set='01',pred='01',time='all',tid=3,kernxy=5,kernz=3,)
  config.trainer.sampler = data_and_sampling.structN2V_sampler
  print(f"Training...")
  T = detect_isbi.init(config.trainer)
  detect_isbi.train(T,config.trainer)
  
  for pset in ['01','02']:
    config = run_isbi.celegans_isbi(train_set='01',pred=pset,time='all',tid=2,kernxy=5,kernz=3,)
    
    print(f"Predicting on {pset}")
    predict.isbi_predict(config.predictor)
    
    print(f"Evaluating on {pset}")
    predict.total_matches(config.evaluator)
    predict.rasterize_isbi_detections(config.evaluator)
    predict.evaluate_isbi_DET(config.evaluator)



def runme():
  config = run_isbi.celegans_isbi(train_set='01',pred='01',time='all',tid=3,kernxy=3,kernz=1,)
  config.trainer.sampler = 'flat'
  print(f"Training...")
  T = detect_isbi.init(config.trainer)
  detect_isbi.train(T,config.trainer)
  
  for pset in ['01','02']:
    config = run_isbi.celegans_isbi(train_set='01',pred=pset,time='all',tid=3,kernxy=3,kernz=1,)
    
    print(f"Predicting on {pset}")
    predict.isbi_predict(config.predictor)
    
    print(f"Evaluating on {pset}")
    predict.total_matches(config.evaluator)
    predict.rasterize_isbi_detections(config.evaluator)
    predict.evaluate_isbi_DET(config.evaluator)