package Models.Robot.LUTRobot;

import Models.LUT.StateActionTable;
import Tools.LogFile;
import java.awt.*;
import java.util.Random;

import robocode.*;
import static robocode.util.Utils.normalRelativeAngleDegrees;


public class LUT_RobotRunner extends AdvancedRobot {

    // Define the States and Actions for each category
    public enum enumEnergy {zero, dangerous, low, medium, high}             // States for energy
    public enum enumDistance {veryNear, near, normal, far, veryFar}         // States for distance
    public enum enumActions {circle, retreat, advance, goCenter, fire}      // Actions

    // Pick the 5 features (States & Action) for Q-learning
    static private StateActionTable stateActionTable5 = new StateActionTable(
            enumEnergy.values().length,         // Our HP
            enumDistance.values().length,       // Distance to enemy
            enumEnergy.values().length,         // Enemy's HP
            enumDistance.values().length,       // Distance to the field center
            enumActions.values().length         // Our actions
    );

    // Initialize current and previous States & Action
    private enumEnergy currMyEnergy = enumEnergy.high;
    private enumEnergy currEnemyEnergy = enumEnergy.high;
    private enumDistance currDistanceToEnemy = enumDistance.near;
    private enumDistance currDistanceToCenter = enumDistance.near;
    private enumActions currAction = enumActions.circle;

    private enumEnergy prevMyEnergy = enumEnergy.high;
    private enumEnergy prevEnemyEnergy = enumEnergy.high;
    private enumDistance prevDistanceToEnemy = enumDistance.near;
    private enumDistance prevDistanceToCenter = enumDistance.near;
    private enumActions prevAction = enumActions.circle;

    // Set the hyperparameters
    private final boolean IS_ONPOLICY = false;      // To implement on-policy or off-policy learning
    private final int EXPLORING_EPISODES = 1000;    // Total episodes before epsilon = 0
    // (no exploration later)

    private final double LEARNING_RATE = 0.5;       // ALPHA: Learning rate
    private final double DISCOUNT_RATE  = 0.8;      // GAMMA: Discount rate
    private double epsilon = 0.75;                  // Epsilon: Explore rate

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
    static String LUT_Filename =  "LUTRobot_StateActionTable.txt";
    static String LOG_Filename = "LUTRobot_Statistics.txt";
    static LogFile log = null;

    // Initialize statistic parameters
    static int totalNumRounds = 0;
    static int numRoundsTo100 = 0;
    static int numWins = 0;
    static boolean isWin;

    // Initialize the location of the battlefield center
    int xMid;
    int yMid;


    // Main method of the robot, operations should be in this section
    @Override
    public void run() {
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
            log.stream.printf("LEARNING RATE (ALPHA), %2.2f\n", LEARNING_RATE);
            log.stream.printf("DISCOUNT RATE (GAMMA), %2.2f\n", DISCOUNT_RATE);
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
        if (Math.random() < epsilon) {
            // Exploration
            currAction = getRandomAction();
        } else {
            // Exploitation
            currAction = getBestAction();
        }

        // Turn body and move robot will be conducted simultaneously (AdvancedRobot)
        switch (currAction) {
            // Circle the enemy
            case circle: {
                setTurnRight(enemyBearingValue + 90);
                setAhead(40);
                break;
            }
            // Turn back with a curving trajectory
            case retreat: {
                setTurnRight(enemyBearingValue + 135);
                setBack(50);
                break;
            }
            // Rush to the enemy
            case advance: {
                setTurnRight(enemyBearingValue);
                setAhead(80);
                break;
            }
            // Head to the center of the battlefield with a curving trajectory
            case goCenter: {
                double bearing = getBearingToCenter(getX(), getY(), xMid, yMid, getHeadingRadians());
                setTurnRight(bearing);
                setAhead(80);
                break;
            }
            // Turn the gun while firing (maximum power is 3 per shot)
            case fire: {
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

        // Update Q-value for previous states & action (t)
        double[] prevStateActionIndex = getPrevStateActionIndex();
        stateActionTable5.train(prevStateActionIndex, computeQ(currReward));
    }


    // Input current total reward to compute the Q-value of previous states & action (t)
    public double computeQ(double reward) {
        /** Determine on-policy or off-policy.
         *  Follow different policy to pick the next Action (t+1)
         *  for computing the Q-value of previous states & action (t).
         */
        enumActions nextAction;

        if (IS_ONPOLICY) {
            // On-policy: Use the epsilon-greedy policy to pick next action (t+1) for c Q-value
            if (Math.random() < epsilon) {
                // Exploration
                nextAction = getRandomAction();
            } else {
                // Exploitation
                nextAction = getBestAction();
            }
        } else {
            // Off-policy: Use the best next action (t+1) for computing Q-value
            nextAction = getBestAction();
        }

        // Get the array of index for previous states & action (t)
        double[] prevStateActionIndex = getPrevStateActionIndex();

        // Get the array of index for current states & action (t+1)
        double[] currStateActionIndex = new double[]{
                currMyEnergy.ordinal(),
                currDistanceToEnemy.ordinal(),
                currEnemyEnergy.ordinal(),
                currDistanceToCenter.ordinal(),
                nextAction.ordinal()
        };

        // Get the Q-value for previous and current states & action
        double prevQ = stateActionTable5.outputFor(prevStateActionIndex);
        double currQ = stateActionTable5.outputFor(currStateActionIndex);

        // Compute and return the NEW Q-VALUE for previous states & action (t)
        return prevQ + LEARNING_RATE * (reward + DISCOUNT_RATE * currQ - prevQ);
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
        stateActionTable5.train(prevStateActionIndex, computeQ(currReward));

        // Update the win rate for each batch to the log file
        isWin = true;
        recordLog(isWin);

        // Save and update the LUT file after Q-learning
        stateActionTable5.save(getDataFile(LUT_Filename));
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
        stateActionTable5.train(prevStateActionIndex, computeQ(currReward));

        // Update the win rate for each batch to the log file
        isWin = false;
        recordLog(isWin);

        // Save and update the LUT file after Q-learning
        stateActionTable5.save(getDataFile(LUT_Filename));
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
        if (numRoundsTo100 < 100) {
            numRoundsTo100++;
            totalNumRounds++;
            // Add a win only for winning an episode
            if (winEpisode) {
                numWins++;
            }
        } else {
            // Record the win rate for each batch (100 episodes)
            log.stream.printf("%d - %d  win rate, %2.1f\n", totalNumRounds - 100, totalNumRounds, 100.0 * numWins / numRoundsTo100);
            log.stream.flush();
            // Set for the next batch
            numRoundsTo100 = 0;
            numWins = 0;
        }
    }


    // Our robot's bullet hits the enemy (gain GOOD intermediate reward)
    @Override
    public void onBulletHit(BulletHitEvent e) {
        currReward += goodIntermediateReward;
    }


    // Enemy's bullet hits our robot (gain BAD intermediate reward)
    @Override
    public void onHitByBullet(HitByBulletEvent e) {
        currReward += badIntermediateReward;
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
    public enumActions getRandomAction() {
        Random rand = new Random();
        int r = rand.nextInt(enumActions.values().length);
        return enumActions.values()[r];
    }


    // Get the best action
    public enumActions getBestAction() {
        return selectBestAction(
                myEnergyValue,
                enemyDistanceValue,
                enemyEnergyValue,
                distanceToCenter(myX, myY, xMid, yMid)
        );
    }


    // Input the STATE VALUES, and will return the best action
    public enumActions selectBestAction(double e, double d, double e2, double d2) {
        // Get the index of each state
        int energyStateIndex = enumEnergyOf(e).ordinal();
        int distanceStateIndex = enumDistanceOf(d).ordinal();
        int enemyEnergyStateIndex = enumEnergyOf(e2).ordinal();
        int distanceToCenterStateIndex = enumDistanceOf(d2).ordinal();
        // Initialize the best Q-value (set to the smallest value)
        double bestQ = -Double.MAX_VALUE;
        enumActions bestAction = null;
        // Find the Q-value (highest) for the best action
        for (int actionIndex = 0; actionIndex < enumActions.values().length; actionIndex++) {
            // Get the index array (state & action) for each Action
            double[] stateActionIndex = new double[] {
                    energyStateIndex,
                    distanceStateIndex,
                    enemyEnergyStateIndex,
                    distanceToCenterStateIndex,
                    actionIndex
            };
            // Compare the Q-value of each Action, and get the best action
            if (stateActionTable5.outputFor(stateActionIndex) > bestQ) {
                // Update Q-value for finding another higher Q-value
                bestQ = stateActionTable5.outputFor(stateActionIndex);
                // Get the best action
                bestAction = enumActions.values()[actionIndex];
            }
        }
        return bestAction;
    }


    // Define the distance state
    public enumDistance enumDistanceOf(double distance) {
        enumDistance distanceState = null;
        if (distance < 75) {
            // For veryNear state
            distanceState = enumDistance.veryNear;
        } else if (distance >= 75 && distance < 200) {
            // For near state
            distanceState = enumDistance.near;
        } else if (distance >= 200 && distance < 500) {
            // For normal state
            distanceState = enumDistance.normal;
        } else if (distance >= 500 && distance < 700) {
            // For far state
            distanceState = enumDistance.far;
        } else if (distance >= 700) {
            // For veryFar state
            distanceState = enumDistance.veryFar;
        }
        return distanceState;
    }


    // Define the energy state
    public enumEnergy enumEnergyOf(double energy) {
        enumEnergy energyState = null;
        if (energy == 0) {
            // For zero state
            energyState = enumEnergy.zero;
        } else if (energy > 0 && energy < 15) {
            // For dangerous state
            energyState = enumEnergy.dangerous;
        } else if (energy >= 15 && energy < 35) {
            // For low state
            energyState = enumEnergy.low;
        } else if (energy >= 35 && energy < 65) {
            // For medium state
            energyState = enumEnergy.medium;
        } else if (energy >= 65) {
            // For high state
            energyState = enumEnergy.high;
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

}
