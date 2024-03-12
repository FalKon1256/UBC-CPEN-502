package Models.Robot.SimpleDQNRobot;

import Models.NeuralNet.NN_OneHiddenLayer;
import Models.ReplayMemory.ReplayMemory;
import Models.Robot.LUTRobot.LUT_RobotRunner;
import Tools.LogFile;
import robocode.*;

import java.awt.*;
import java.io.File;
import java.util.Random;

import static Models.NeuralNet.NN_OneHiddenLayer.fixedWeightMax;
import static Models.NeuralNet.NN_OneHiddenLayer.fixedWeightMin;
import static robocode.util.Utils.normalRelativeAngleDegrees;


public class SimpleDQN_RobotRunner extends AdvancedRobot {

     // Define the States and Actions for each category
     public enum enumEnergy {zero, dangerous, low, medium, high}             // States for energy
     public enum enumDistance {veryNear, near, normal, far, veryFar}         // States for distance
     public enum enumActions {circle, retreat, advance, goCenter, fire}      // Actions

     // Initialize current and previous States & Action
     private LUT_RobotRunner.enumEnergy currMyEnergy = LUT_RobotRunner.enumEnergy.high;
     private LUT_RobotRunner.enumEnergy currEnemyEnergy = LUT_RobotRunner.enumEnergy.high;
     private LUT_RobotRunner.enumDistance currDistanceToEnemy = LUT_RobotRunner.enumDistance.near;
     private LUT_RobotRunner.enumDistance currDistanceToCenter = LUT_RobotRunner.enumDistance.near;
     private LUT_RobotRunner.enumActions currAction = LUT_RobotRunner.enumActions.circle;

     private LUT_RobotRunner.enumEnergy prevMyEnergy = LUT_RobotRunner.enumEnergy.high;
     private LUT_RobotRunner.enumEnergy prevEnemyEnergy = LUT_RobotRunner.enumEnergy.high;
     private LUT_RobotRunner.enumDistance prevDistanceToEnemy = LUT_RobotRunner.enumDistance.near;
     private LUT_RobotRunner.enumDistance prevDistanceToCenter = LUT_RobotRunner.enumDistance.near;
     private LUT_RobotRunner.enumActions prevAction = LUT_RobotRunner.enumActions.circle;

     int currActionNum;

     // NN structure
     private static final int INPUT_LAYERS_NUM = 4;
     private static final int HIDDEN_LAYERS_NUM = 10;
     private static final int OUTPUT_LAYERS_NUM = 1;
     public static NN_OneHiddenLayer[] nn = new NN_OneHiddenLayer[enumActions.values().length];
     private double RMSError = 0.0;

     // NN training hyperparameters
     private static final double NN_LEARNING_RATE = 0.4;
     private static final double NN_MOMENTUM = 0.8;

     // Set the hyperparameters
     private final boolean IS_ONPOLICY = false;      // To implement on-policy or off-policy learning
     private final int EXPLORING_EPISODES = 1000;    // Total episodes before epsilon = 0
     // (no exploration later)

     private final double Q_LEARNING_RATE = 0.1;       // ALPHA: Learning rate
     private final double Q_DISCOUNT_RATE  = 0.8;      // GAMMA: Discount rate
     private static double epsilon = 0.75;                  // Epsilon: Explore rate

     // Set the Rewards
     private final double goodTerminalReward = 1.0;          // Good reward of winning one episode
     private final double badTerminalReward = -1.0;          // Bad reward of losing one episode
     private final double goodIntermediateReward = 0.5;      // Good reward within each episode
     private final double badIntermediateReward = -0.25;     // Bad reward within each episode
     private double currReward = 0.0;                        // Record the current reward for each episode

     // Initialize current State Values
     double myX = 0.0;
     double myY = 0.0;
     double myEnergyValue = 0.0;
     double enemyBearingValue = 0.0;
     double enemyDistanceValue = 0.0;
     double enemyEnergyValue = 0.0;

     // Set logging parameters
     private static String scoreListFile = "scoreList_LR_" + NN_LEARNING_RATE + "_MT_" + NN_MOMENTUM + "_HidNum_" + HIDDEN_LAYERS_NUM + "_epsilon_" + epsilon + ".txt";
     static String LOG_Filename = "NNRobot_Statistics.txt";
     static LogFile log = null;
     private File weights[] = new File[enumActions.values().length];
     private static final int BATCH = 100;

     // Initialize statistic parameters
     static int totalNumRounds = 0;
     static int numRoundsTo100 = 0;
     static int numWins = 0;
     static boolean isWin;

     // Initialize the location of the battlefield center
     int xMid;
     int yMid;

     // Initialize Replay Memory parameters
     private static final boolean RECORD_MEMORY_ON = false;
     private static final int RECORD_MEMORY_N = 15;
     public static ReplayMemory<Experience> memory = new ReplayMemory<>(RECORD_MEMORY_N);

     // Initialize Experience
     public static class Experience {
          double[] currState;
          int action;
          double reward;
          double[] nextState;
     }


     // Main method of the robot, operations should be in this section
     @Override
     public void run() {

          // Create a new NN for training
          setNeuralNets();

          // Load previous NN weights
          loadWeights();

          // Set our robot style
          setBodyColor(Color.black);
          setGunColor(Color.darkGray);
          setRadarColor(Color.white);
          setBulletColor(Color.magenta);
          setScanColor(Color.green);

          // Get the location of the battlefield center
          xMid = (int) getBattleFieldWidth() / 2;
          yMid = (int) getBattleFieldHeight() / 2;

          // Create the log file for statistics
          if (log == null) {
               log = new LogFile(getDataFile(LOG_Filename));
               log.stream.print("----------HYPERPARAMETERS----------\n");
               log.stream.printf("ON-POLICY, %s\n", IS_ONPOLICY ? "TRUE":"FALSE");
               log.stream.printf("EXPLORING EPISODES, %d\n", EXPLORING_EPISODES);
               log.stream.printf("LEARNING RATE (ALPHA), %2.2f\n", Q_LEARNING_RATE);
               log.stream.printf("DISCOUNT RATE (GAMMA), %2.2f\n", Q_DISCOUNT_RATE);
               log.stream.printf("EXPLORE RATE (EPSILON), %2.2f\n\n", epsilon);
               log.stream.print("--------------REWARDS--------------\n");
               log.stream.printf("GOOD TERMINAL REWARD, %2.2f\n", goodTerminalReward);
               log.stream.printf("BAD TERMINAL REWARD, %2.2f\n", badTerminalReward);
               log.stream.printf("GOOD INTERMEDIATE REWARD, %2.2f\n", goodIntermediateReward);
               log.stream.printf("BAD INTERMEDIATE REWARD, %2.2f\n\n", badIntermediateReward);
          }

          // Robot runs the actions for each turn (each loop)
          while (true) {
               // Set epsilon to 0 (exploitation for every move) after the learning episodes
               if (totalNumRounds > EXPLORING_EPISODES) {
                    epsilon = 0;
               }
               // Robot picks the current action
               robotAction();
               // Robot sets the radar action
               radarAction();
               /** Robot will conduct all actions for this turn (AdvancedRobot needs an explicit order).
                *  Robot body, gun, and radar actions can be conducted simultaneously.
                */
               execute();
          }
     }


     // Pick the current action of our robot
     private void robotAction() {

          // Get current action
          currActionNum = getAction();

          // Turn body and move robot will be conducted simultaneously (AdvancedRobot)
          switch (currActionNum) {
               // Circle the enemy
               case 0: {
                    setTurnRight(enemyBearingValue + 90);
                    setAhead(40);
                    break;
               }
               // Turn back with a curving trajectory
               case 1: {
                    setTurnRight(enemyBearingValue + 135);
                    setBack(50);
                    break;
               }
               // Rush to the enemy
               case 2: {
                    setTurnRight(enemyBearingValue);
                    setAhead(80);
                    break;
               }
               // Head to the center of the battlefield with a curving trajectory
               case 3: {
                    double bearing = getBearingToCenter(getX(), getY(), xMid, yMid, getHeadingRadians());
                    setTurnRight(bearing);
                    setAhead(80);
                    break;
               }
               // Turn the gun while firing (maximum power is 3 per shot)
               case 4: {
                    turnGunRight(normalRelativeAngleDegrees(getHeading() - getGunHeading() + enemyBearingValue));
                    setFire(3);
                    break;
               }
          }
     }


     // Set the radar action
     private void radarAction() {
          // Radar spins 360 degrees each turn (radar always spinning)
          setTurnRadarRightRadians(2 * Math.PI);
     }


     public void setNeuralNets(){
          for(int i = 0; i< enumActions.values().length; i++){
               nn[i]=new NN_OneHiddenLayer(INPUT_LAYERS_NUM, HIDDEN_LAYERS_NUM, OUTPUT_LAYERS_NUM, NN_LEARNING_RATE, NN_MOMENTUM, fixedWeightMin, fixedWeightMax, true);
          }
     }


     // Get the next action
     public int getAction() {

          // Save the current and previous state
          double[] currStates = new double[5];
          currStates[0] = currMyEnergy.ordinal();
          currStates[1] = currDistanceToEnemy.ordinal();
          currStates[2] = currEnemyEnergy.ordinal();
          currStates[3] = currDistanceToCenter.ordinal();

          double[] prevStates = new double[5];
          prevStates[0] = prevMyEnergy.ordinal();
          prevStates[1] = prevDistanceToEnemy.ordinal();
          prevStates[2] = prevEnemyEnergy.ordinal();
          prevStates[3] = prevDistanceToCenter.ordinal();

          // Get the current and previous best action
          int nextAction = getBestAction(currStates);
          int prevAction = getBestAction(prevStates);

          // Get the current and previous Q-values
          double currQ = nn[nextAction].outputFor(currStates);
          double prevQ = nn[prevAction].outputFor(prevStates);

          // Calculate the error
          double error = Q_LEARNING_RATE * (currReward + Q_DISCOUNT_RATE * currQ - prevQ);
          RMSError += error * error;

          // Train the weights of NN
          double correctPrevQ = prevQ + error;
          nn[prevAction].train(prevStates, correctPrevQ);

          // Train replay memory
          if(RECORD_MEMORY_ON) {
               // Create and save the experiences
               Experience exp = new Experience();
               exp.currState = prevStates;
               exp.action = prevAction;
               exp.reward = currReward;
               exp.nextState = currStates;
               memory.add(exp);
          }

          // Keep the current states
          for (int i = 0; i < INPUT_LAYERS_NUM; i++) {
               prevStates[i] = currStates[i];
          }

          // Exploration or exploitation
          if(Math.random() < epsilon) {
               return (int)(Math.random() * enumActions.values().length);
          }
          return nextAction;
     }

     // Get the best action
     public int getBestAction(double[] currState) {
          int nextAction = 0;
          for(int i = 0; i < enumActions.values().length; i++) {
               if(nn[i].outputFor(currState) > nn[nextAction].outputFor(currState)) {
                    nextAction = i;
               }
          }
          return nextAction;
     }

     /** When enemy is scanned by our radar each turn:
      *  1. Update the States (previous and current) and Action (only previous).
      *  2. The picked Action has been executed, so the States and Action should be "previous" (t to t+1).
      *  3. Update the Q-value for "previous states & action" (t).
      */
     @Override
     public void onScannedRobot(ScannedRobotEvent e) {
          // Get the current state values
          myX = getX();
          myY = getY();
          myEnergyValue = getEnergy();
          enemyDistanceValue = e.getDistance();
          enemyEnergyValue = e.getEnergy();
          enemyBearingValue = e.getBearing();

          // Update States and Action (t to t+1)
          prevMyEnergy = currMyEnergy;
          prevDistanceToEnemy = currDistanceToEnemy;
          prevEnemyEnergy = currEnemyEnergy;
          prevDistanceToCenter = currDistanceToCenter;
          prevAction = currAction;

          currMyEnergy = enumEnergyOf(myEnergyValue);
          currDistanceToEnemy = enumDistanceOf(enemyDistanceValue);
          currEnemyEnergy = enumEnergyOf(enemyEnergyValue);
          currDistanceToCenter = enumDistanceOf(distanceToCenter(myX, myY, xMid, yMid));

     }


     /** Our robot WINS the episode:
      *  1. Update current rewards.
      *  2. Update Q-value.
      *  3. Update win rate for each batch (100 episodes)
      *  4. Save and update the LUT file after Q-learning (update at the end of each episode)
      */
     @Override
     public void onWin(WinEvent e) {
          // Gain GOOD terminal reward for winning an episode
          currReward += goodTerminalReward;

          // Update Q-value (need to do this since our robot cannot scan after the end of the game)
          double[] prevStateActionIndex = getPrevStateActionIndex();
          getAction();

          // Save the win rate for each batch to the log file
          isWin = true;
          recordLog(isWin);
          saveWeights();
     }


     /** Our robot LOSES the episode:
      *  1. Update current rewards.
      *  2. Update Q-value.
      *  3. Update win rate for each batch (100 episodes)
      *  4. Save and update the LUT file after Q-learning (update at the end of each episode)
      */
     @Override
     public void onDeath(DeathEvent e) {
          // Gain BAD terminal reward for winning an episode
          currReward += badTerminalReward;

          // Update Q-value (need to do this since our robot cannot scan after the end of the game)
          double[] prevStateActionIndex = getPrevStateActionIndex();
          getAction();

          // Save the win rate for each batch to the log file
          isWin = false;
          recordLog(isWin);
          saveWeights();
     }


     // Returns the array of indices for previous States & Action
     public double[] getPrevStateActionIndex() {
          return new double[] {
                  prevMyEnergy.ordinal(),
                  prevDistanceToEnemy.ordinal(),
                  prevEnemyEnergy.ordinal(),
                  prevDistanceToCenter.ordinal(),
                  prevAction.ordinal()
          };
     }


     // Update the win rate for each batch (100 episodes) to the log file
     public void recordLog(boolean winEpisode) {
          if (numRoundsTo100 < BATCH) {
               numRoundsTo100++;
               totalNumRounds++;
               // Add a win only for winning an episode
               if (winEpisode) {
                    numWins++;
               }
          } else {

               try{
                    File file = getDataFile(scoreListFile);
                    RobocodeFileWriter fileWriter = new RobocodeFileWriter(file.getAbsolutePath(), true);
                    RMSError = Math.sqrt(RMSError/625);
                    fileWriter.write(String.format("%d - %d  win rate: %2.1f, RMS error: %s\n", totalNumRounds - BATCH, totalNumRounds, 100.0 * numWins / numRoundsTo100, RMSError));
                    fileWriter.close();
               } catch(Exception e) {
                    System.out.println(e);
               }

               // Set for the next batch
               numRoundsTo100 = 0;
               numWins = 0;
          }
     }


     // Our robot's bullet hits the enemy (gain GOOD intermediate reward)
     @Override
     public void onBulletHit(BulletHitEvent e) {
          currReward += goodIntermediateReward;
          replayMemoryTraining();
     }


     // Enemy's bullet hits our robot (gain BAD intermediate reward)
     @Override
     public void onHitByBullet(HitByBulletEvent e) {
          currReward += badIntermediateReward;
          replayMemoryTraining();
     }


     // Our robot collides with the wall (gain BAD intermediate reward), robot's HP will decrease
     @Override
     public void onHitWall(HitWallEvent e) {
          super.onHitWall(e);
          currReward += badIntermediateReward;
          avoidObstacle();
     }


     // Our robot collides with the enemy, both robots' HP will decrease
     @Override
     public void onHitRobot(HitRobotEvent e) {
          super.onHitRobot(e);
          avoidObstacle();
     }


     // Robot's action to avoid the wall or enemy (after colliding)
     public void avoidObstacle() {
          switch (currAction) {
               case circle: {
                    setBack(20);
                    break;
               }
               case retreat: {
                    setTurnRight(45);
                    setAhead(25);
                    execute();
                    break;
               }
               case advance: {
                    setTurnRight(45);
                    setBack(40);
                    execute();
                    break;
               }
          }
     }


     // Get a random action
     public LUT_RobotRunner.enumActions getRandomAction() {
          Random rand = new Random();
          int r = rand.nextInt(LUT_RobotRunner.enumActions.values().length);
          return LUT_RobotRunner.enumActions.values()[r];
     }


     // Define the distance state
     public LUT_RobotRunner.enumDistance enumDistanceOf(double distance) {
          LUT_RobotRunner.enumDistance distanceState = null;
          if (distance < 75) {
               // For veryNear state
               distanceState = LUT_RobotRunner.enumDistance.veryNear;
          } else if (distance >= 75 && distance < 200) {
               // For near state
               distanceState = LUT_RobotRunner.enumDistance.near;
          } else if (distance >= 200 && distance < 500) {
               // For normal state
               distanceState = LUT_RobotRunner.enumDistance.normal;
          } else if (distance >= 500 && distance < 700) {
               // For far state
               distanceState = LUT_RobotRunner.enumDistance.far;
          } else if (distance >= 700) {
               // For veryFar state
               distanceState = LUT_RobotRunner.enumDistance.veryFar;
          }
          return distanceState;
     }


     // Define the energy state
     public LUT_RobotRunner.enumEnergy enumEnergyOf(double energy) {
          LUT_RobotRunner.enumEnergy energyState = null;
          if (energy == 0) {
               // For zero state
               energyState = LUT_RobotRunner.enumEnergy.zero;
          } else if (energy > 0 && energy < 15) {
               // For dangerous state
               energyState = LUT_RobotRunner.enumEnergy.dangerous;
          } else if (energy >= 15 && energy < 35) {
               // For low state
               energyState = LUT_RobotRunner.enumEnergy.low;
          } else if (energy >= 35 && energy < 65) {
               // For medium state
               energyState = LUT_RobotRunner.enumEnergy.medium;
          } else if (energy >= 65) {
               // For high state
               energyState = LUT_RobotRunner.enumEnergy.high;
          }
          return energyState;
     }


     // Returns the distance between our robot and the battlefield center
     public double distanceToCenter(double fromX, double fromY, double toX, double toY) {
          return Math.sqrt(Math.pow((fromX - toX), 2) + Math.pow((fromY - toY), 2));
     }


     // Returns the radian that our robot body has to turn to face the battlefield center
     public double getBearingToCenter(double fromX, double fromY, double toX, double toY, double currHeadingRadian) {
          // Get the radian between +Y axis and the input vector (clockwise is positive)
          double inputVectorRadian = (Math.PI / 2) - Math.atan2(toY - fromY, toX - fromX);
          return convertRadian(inputVectorRadian - currHeadingRadian);
     }


     // Convert input radian to the range of [-Pi, Pi]
     public double convertRadian(double radian) {
          // Keep converting until the radian is between -Pi and Pi
          while (radian <= -Math.PI) {
               radian += 2 * Math.PI;
          }
          while (radian > Math.PI) {
               radian -= 2 * Math.PI;
          }
          return radian;
     }


     // Replay memory Training
     public void replayMemoryTraining() {
          // Create experience slots
          Object[] experiences = memory.randomSample(RECORD_MEMORY_N);
          // Train experiences
          for(Object e: experiences) {
               // Cast the data type
               Experience x = (Experience) e;
               // Get the best action
               int nextAction = getBestAction(x.currState);
               // Get the Q-values and calculate the error
               double currentQ = nn[nextAction].outputFor(x.nextState);
               double previousQ = nn[x.action].outputFor(x.currState);
               double error = Q_LEARNING_RATE * (x.reward + Q_DISCOUNT_RATE * currentQ - previousQ);
               // Train the NN weights for replay memory
               double correctPreviousQ = previousQ + error;
               nn[currActionNum].train(x.currState, correctPreviousQ);
          }
     }


     // Save all NN weights
     public void saveWeights() {
          for(int i = 0; i < enumActions.values().length; i++) {
               String fileName = "weights" + i + ".txt";
               // Save the NN weights for each action
               weights[i] = getDataFile(fileName);
               nn[i].saveWeights(weights[i]);
          }
     }


     // Load all NN weights
     public void loadWeights() {
          for(int i = 0; i < enumActions.values().length; i++) {
               String fileName = "weights" + i + ".txt";
               // Load the NN weights for each action
               weights[i] = getDataFile(fileName);
               nn[i].loadWeights(weights[i]);
          }
     }

}

