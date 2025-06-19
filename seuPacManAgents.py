# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState
from multiAgents import MultiAgentSearchAgent


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        """
        Determina a melhor ação para o Pac-Man usando a estratégia Minimax.
        """

        def minimax(agentIndex, depth, state):
            # Condição de parada: fim de jogo ou profundidade máxima
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            totalAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % totalAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0:
                # Pac-Man (maximizador)
                maxValue = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    nextState = state.generateSuccessor(agentIndex, action)
                    value = minimax(nextAgent, nextDepth, nextState)
                    if value > maxValue:
                        maxValue = value
                return maxValue
            else:
                # Fantasma (minimizador)
                minValue = float('inf')
                for action in state.getLegalActions(agentIndex):
                    nextState = state.generateSuccessor(agentIndex, action)
                    value = minimax(nextAgent, nextDepth, nextState)
                    if value < minValue:
                        minValue = value
                return minValue

        # Ação ideal para Pac-Man na raiz da árvore
        bestScore = float('-inf')
        optimalAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, successor)
            if score > bestScore:
                bestScore = score
                optimalAction = action

        return optimalAction


def betterEvaluationFunction(currentGameState: GameState):
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()

    # Calcula a distância de Manhattan para a comida mais próxima
    foodDistances = [manhattanDistance(pos, f) for f in food]
    if len(foodDistances) > 0:
        minFoodDistance = min(foodDistances)
    else:
        minFoodDistance = 0

    # Distância para o fantasma mais próximo
    ghostDistances = [manhattanDistance(pos, ghost.getPosition()) for ghost in ghostStates]
    minGhostDistance = min(ghostDistances)

    # Aumenta a pontuação se o fantasma estiver assustado, mas penaliza se estiver muito perto
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    if min(scaredTimes) > 0:
        minGhostDistance = 0  # Ignora fantasmas assustados

    return currentGameState.getScore() - (1.5 / (minFoodDistance + 1)) + (2 / (minGhostDistance + 1))


# Abbreviation
better = betterEvaluationFunction
