# riichi_mahjong
Create a riichi mahjong AI to play on tenhou.net

## Evaluation of waiting tiles prediction

Number of effective samples: 75110

Evaluation value: 0.8220610355708976

## Evaluation of HS prediction

(1). HS

     MSE =  0.46887228095040395
     
(2). HS_WFW: 

     MSE = 0.3700624104298077

## Next Job

Finish Monte Carlo simulation.

## **Note**

(1). Opponent model (mahjong.ai.opponent_model.py) is now under development. Note that we have already switched to opponent model in mahjong.player.py (line 262-266), which means we have intentionally turn off the original AI of the author.
