"""This module contains the UE, Packet Flow, Packet, PcktQueue, Bearer and RadioLink classes.
These classes are oriented to describe UE traffic profile, and UE relative concepts.
"""
import os
import random
import sys
from collections import deque
from scipy.stats import truncnorm, lognorm
from scipy.stats import truncexpon
from scipy.stats import uniform
import numpy as np
import simpy


# UE class: terminal description
class UE:
    """This class is used to model UE behaviour and relative properties"""

    def __init__(self, i, ue_sinr0, p, npM):
        self.id = i
        self.state = "RRC-IDLE"
        self.packetFlows = []
        self.bearers = []
        self.radioLinks = RadioLink(1, ue_sinr0, self.id)
        self.TBid = 1
        self.pendingPckts = {}
        self.prbs = p
        self.resUse = 0
        self.pendingTB = []
        self.bler = 0
        self.tbsz = 1
        self.MCS = 0
        self.pfFactor = 1  # PF Scheduler
        self.pastTbsz = deque([1])  # PF Scheduler
        self.lastDen = 0.001  # PF Scheduler
        self.num = 0  # PF Scheduler
        self.BWPs = npM
        self.TXedTB = 1
        self.lostTB = 0
        self.symb = 0
        self.traffic_s = 0  # apex.py
        self.auxBuff = deque([])  # apex.py
        self.satisfied = False
        self.delay = 0
        self.throughput = 0
        self.sndbytes = 0
        self.utilization_list = []
        self.utilization = 0
        self.pks_s = 0
        self.tbSize = 0

    def insertPckt(self, p):
        self.auxBuff.append(p)

    def insertPcktLeft(self, p):
        self.auxBuff.appendleft(p)

    def removePckt(self):
        if len(self.auxBuff) > 0:
            return self.auxBuff.popleft()

    def addPacketFlow(self, pckFl):
        self.packetFlows.append(pckFl)

    def addBearer(self, br):
        self.bearers.append(br)

    def receivePckt(self, env, c):  # PEM -------------------------------------------
        """This method takes packets on the application buffers and leave them on the bearer buffers. This is a PEM method."""
        while True:
            if len(self.packetFlows[0].appBuff.pckts) > 0:
                if self.state == "RRC-IDLE":  # Not connected
                    self.connect(c)
                    nextPackTime = c.tUdQueue
                    yield env.timeout(nextPackTime)
                    if nextPackTime > c.inactTimer:
                        self.releaseConnection(c)
                else:  # Already connected user
                    self.queueDataPckt(c)
                    nextPackTime = c.tUdQueue
                    yield env.timeout(nextPackTime)
                    if nextPackTime > c.inactTimer:
                        self.releaseConnection(c)
            else:
                nextPackTime = c.tUdQueue
                yield env.timeout(nextPackTime)

    def connect(self, cl):
        """This method creates bearers and bearers buffers."""
        bD = Bearer(1, 9, self.packetFlows[0].type)
        self.addBearer(bD)
        self.queueDataPckt(cl)
        if self.packetFlows[0].type == "DL":
            if (
                list(
                    cl.interSliceSched.slices[
                        self.packetFlows[0].sliceName
                    ].schedulerDL.ues.keys()
                ).count(self.id)
            ) < 1:
                cl.interSliceSched.slices[
                    self.packetFlows[0].sliceName
                ].schedulerDL.ues[self.id] = self
        else:
            if (
                list(
                    cl.interSliceSched.slices[
                        self.packetFlows[0].sliceName
                    ].schedulerUL.ues.keys()
                ).count(self.id)
            ) < 1:
                cl.interSliceSched.slices[
                    self.packetFlows[0].sliceName
                ].schedulerUL.ues[self.id] = self
        self.state = "RRC-CONNECTED"

    def queueDataPckt(self, cell):
        """This method queues the packets taken from the application buffer in the bearer buffers."""
        pD = self.packetFlows[0].appBuff.removePckt()
        buffSizeAllUEs = 0
        buffSizeThisUE = 0
        if self.packetFlows[0].type == "DL":
            for ue in list(
                cell.interSliceSched.slices[
                    self.packetFlows[0].sliceName
                ].schedulerDL.ues.keys()
            ):
                buffSizeUE = 0
                for p in (
                    cell.interSliceSched.slices[self.packetFlows[0].sliceName]
                    .schedulerDL.ues[ue]
                    .bearers[0]
                    .buffer.pckts
                ):
                    buffSizeUE = buffSizeUE + p.size
                if self.id == ue:
                    buffSizeThisUE = buffSizeUE
                buffSizeAllUEs = buffSizeAllUEs + buffSizeUE
        else:
            for ue in list(
                cell.interSliceSched.slices[
                    self.packetFlows[0].sliceName
                ].schedulerUL.ues.keys()
            ):
                buffSizeUE = 0
                for p in (
                    cell.interSliceSched.slices[self.packetFlows[0].sliceName]
                    .schedulerUL.ues[ue]
                    .bearers[0]
                    .buffer.pckts
                ):
                    buffSizeUE = buffSizeUE + p.size
                if self.id == ue:
                    buffSizeThisUE = buffSizeUE
                buffSizeAllUEs = buffSizeAllUEs + buffSizeUE
        if buffSizeThisUE < cell.maxBuffUE:
            self.bearers[0].buffer.insertPckt(pD)
        else:
            pcktN = pD.secNum
            if self.packetFlows[0].type == "DL":
                cell.interSliceSched.slices[
                    self.packetFlows[0].sliceName
                ].schedulerDL.printDebDataDM(
                    '<p style="color:red"><b>'
                    + str(self.id)
                    + " packet "
                    + str(pcktN)
                    + " lost ....."
                    + str(pD.tIn)
                    + "</b></p>"
                )
            else:
                cell.interSliceSched.slices[
                    self.packetFlows[0].sliceName
                ].schedulerUL.printDebDataDM(
                    '<p style="color:red"><b>'
                    + str(self.id)
                    + " packet "
                    + str(pcktN)
                    + " lost ....."
                    + str(pD.tIn)
                    + "</b></p>"
                )
            self.packetFlows[0].lostPackets = self.packetFlows[0].lostPackets + 1

    def releaseConnection(self, cl):
        self.state = "RRC-IDLE"
        self.bearers = []


# ------------------------------------------------
# PacketFlow class: PacketFlow description
class PacketFlow:
    """This class is used to describe UE traffic profile for the simulation."""

    def __init__(
        self,
        i,
        pckSize,
        pckArrRate,
        u,
        tp,
        slc,
        activationTime,
        deactivationTime,
        distributionSize,
        distributionArrival,
        scheduling_deadline=None,
    ):
        self.id = i
        self.tMed = 0
        self.sMed = 0
        self.type = tp
        self.sliceName = slc
        self.pckArrivalRate = pckArrRate
        self.qosFlowId = 0
        self.packetSize = pckSize
        self.ue = u
        self.sMax = (float(self.packetSize) / 350) * 600
        self.tMax = (float(self.pckArrivalRate) / 6) * 12.5
        self.tMin = (float(self.pckArrivalRate)) * 0.15
        self.tStart = 0
        self.appBuff = PcktQueue()
        self.lostPackets = 0
        self.sentPackets = 0
        self.rcvdBytes = 0
        self.pId = 1
        self.header = 30
        self.meassuredKPI = {"Throughput": 0, "Delay": 0, "PacketLossRate": 0}
        self.activationTime = activationTime
        self.deactivationTime = self.oneOrValue(deactivationTime)
        self.distributionSize = distributionSize
        self.distributionArrival = distributionArrival
        self.schedulingDeadline = scheduling_deadline
        self.deadlineMisses = 0
        self.deadlineEvaluated = 0
        self.schedulingDelays = []
        self.deliveredPackets = 0
        self.deliveredBytes = 0
        self.packetGenerationTimes = {}
        self.completedPacketIds = set()
        self.completionDelays = []

    def recordSchedulingOutcome(self, packet, now):
        """Record first-scheduling delay exactly once per packet."""
        if packet.scheduled_at is not None:
            return
        packet.scheduled_at = now
        packet.scheduling_delay = now - packet.tIn
        self.packetGenerationTimes.setdefault(
            packet.secNum,
            packet.tIn,
        )
        if packet.deadline is None:
            return
        self.deadlineEvaluated += 1
        self.schedulingDelays.append(packet.scheduling_delay)
        packet.deadline_missed = packet.scheduling_delay > packet.deadline
        if packet.deadline_missed:
            self.deadlineMisses += 1

    def recordDeliveryCompletion(
        self,
        packet_id,
        now,
    ):
        """
        Record successful full-packet delivery exactly once.

        Completion delay is measured from packet generation until the
        final successful TB needed for that packet has been received.
        """
        packet_id = int(packet_id)

        if packet_id in self.completedPacketIds:
            return None

        if packet_id not in self.packetGenerationTimes:
            return None

        completion_delay = (
            float(now)
            - float(
                self.packetGenerationTimes[
                    packet_id
                ]
            )
        )

        self.completedPacketIds.add(
            packet_id
        )

        self.completionDelays.append(
            completion_delay
        )

        self.deliveredPackets += 1

        del self.packetGenerationTimes[
            packet_id
        ]

        return completion_delay

    def setQosFId(self, q):
        self.qosFlowId = q

    # function that returns 1 if value is bigger than 1 or the value itself if it is smaller than 1
    def oneOrValue(self, value):
        if value > 1:
            print(
                "Warning: Activation time or deactivation time is bigger than 1. It will be set to 1."
            )
            return 1
        else:
            return value

    # function that returns true if the value is in any of the ranges
    def inRange(self, value, ranges):
        for r in ranges:
            if value in range(r[0], r[1]):
                return True
        return False

    # function that returns true if the value is in any of the continuous ranges
    def inRangeCont(self, value, ranges):
        for r in ranges:
            if value >= r[0] and value <= r[1]:
                return True
        return False

    # multiply a sequence by a float
    def multiply(self, seq, factor):
        return [x * factor for x in seq]

    def queueAppPckt(self, env, tSim):  # --- PEM -----
        """This method creates packets according to the packet flow traffic profile and stores them in the application buffer."""
        ueN = int(self.ue[2:])  # number of UEs in simulation
        self.tStart = random.expovariate(1.0) + tSim * self.activationTime
        yield env.timeout(self.tStart)  # each UE start transmission after tStart
        while env.now < (tSim * self.deactivationTime * 0.83):
            self.sentPackets = self.sentPackets + 1
            size = self.getPsize()
            pD = Packet(self.pId, size + self.header, self.qosFlowId, self.ue)
            self.pId = self.pId + 1
            pD.tIn = env.now
            pD.timestamp = env.now
            pD.deadline = self.schedulingDeadline
            self.appBuff.insertPckt(pD)
            nextPackTime = self.getParrRate()
            yield env.timeout(nextPackTime)

    def truncated_lognormal(self, mean, sigma, maximum, size=1):
        # Calculate parameters for lognormal distribution
        phi = np.sqrt(sigma**2 + mean**2)
        mu = np.log(mean**2 / phi)
        sigma_lognorm = np.sqrt(np.log(phi**2 / mean**2))

        # Generate truncated normal samples
        a = (0 - mean) / sigma
        b = (maximum - mean) / sigma
        normal_samples = np.random.normal(size=size)
        truncated_samples = np.clip(normal_samples, a, b)

        # Transform truncated normal samples to lognormal distribution
        lognormal_samples = np.exp(mu + sigma_lognorm * truncated_samples)

        return lognormal_samples

    def getPsize(self):
        """
        This method returns the size of the next packet to be transmitted.
        The size is determined by the distribution specified in self.distributionSize.
        """
        if self.distributionSize == "Pareto":
            pSize = random.paretovariate(1.2) * (0.2 / 1.2) * 2 + self.packetSize
            return int(pSize)
        elif self.distributionSize == "Pareto2":
            maximum = 700
            alpha = 1.2
            mean = self.packetSize
            size = 1
            k = (alpha - 1) * mean / maximum

            pareto_samples = maximum * (np.random.pareto(alpha, size=size) + k)
            truncated_samples = np.clip(pareto_samples, None, maximum)

            pSize = truncated_samples
            return int(pSize)
        elif self.distributionSize == "Lognormal":
            std = 1
            X = self.packetSize
            mu = np.log(X**2 / np.sqrt(X**2 + std**2))
            sigma = np.sqrt(np.log(1 + (std**2 / X**2)))
            pSize = np.random.lognormal(mu, sigma)

            if pSize > self.sMax:
                pSize = self.sMax

            return int(pSize)
        elif self.distributionSize == "Constant":
            return self.packetSize
        elif self.distributionSize == "Uniform":
            pSize = random.uniform(self.packetSize, self.sMax)
            return int(pSize)
        elif self.distributionSize == "TruncatedNormal":
            pSize = int(self.truncated_lognormal(self.packetSize, 3, 900))
            return pSize
        elif self.distributionSize == "Normal":
            pSize = np.random.normal(self.packetSize, 40)
            if pSize > self.sMax:
                pSize = self.sMax
            return pSize
        elif self.distributionSize == "Uniform2":
            a = 500 - 2
            b = 500 + 2
            return int(uniform.rvs(loc=a, scale=10, size=1)[0])
        elif self.distributionSize == "Normal2":
            pSize = np.random.normal(self.packetSize, self.packetSize / 15)
            if pSize > self.sMax:
                pSize = self.sMax
            return pSize
        elif self.distributionSize == "expon_truncada":
            smax = self.packetSize * 1.1
            smin = self.packetSize * 0.9
            X = self.packetSize
            b = (smax - X) / (X - smin)
            a = smin

            scale = X / (1 - truncexpon.cdf(smax, b=b, loc=0, scale=X))
            return int(truncexpon.rvs(b=b, loc=0, scale=scale, size=1)[0])
        elif self.distributionSize == "Exponential":
            pSize = np.random.exponential(self.packetSize)
            if pSize > self.sMax:
                pSize = self.sMax
            return pSize
        elif self.distributionSize == "Gamma":
            pSize = np.random.gamma(self.packetSize, 0.722)
            if pSize > self.sMax:
                pSize = self.sMax
            return pSize
        elif self.distributionSize == "Weibull":
            pSize = np.random.weibull(self.packetSize)
            while pSize > self.sMax:
                pSize = np.random.weibull(self.packetSize)
            return pSize
        elif self.distributionSize == "Beta":
            pSize = np.random.beta(self.packetSize, 0.722)
            while pSize > self.sMax:
                pSize = np.random.beta(self.packetSize, 0.722)
            return pSize
        else:
            print("Error: Distribution size not defined.")

    def getParrRate(self):
        if self.distributionArrival == "Constant":
            pArrRate = self.pckArrivalRate
        elif self.distributionArrival == "Pareto":
            pArrRate = random.paretovariate(1.2) * (
                self.pckArrivalRate * (0.2 / 1.2)
            )
        elif self.distributionArrival == "Pareto2":
            maximum = 1.4
            alpha = 1.2
            mean = self.pckArrivalRate
            size = 1
            k = (alpha - 1) * mean / maximum

            pareto_samples = maximum * (np.random.pareto(alpha, size=size) + k)
            truncated_samples = np.clip(pareto_samples, None, maximum)
            pArrRate = truncated_samples
            return int(pArrRate)
        elif self.distributionArrival == "Exponential":
            pArrRate = np.random.exponential(self.pckArrivalRate)
            if pArrRate > self.tMax:
                pArrRate = self.tMax
        elif self.distributionArrival == "Uniform":
            pArrRate = random.uniform(0, self.pckArrivalRate)
            if pArrRate > self.tMax:
                pArrRate = self.tMax
        elif self.distributionArrival == "Uniform2":
            pArrRate = random.uniform(0, self.pckArrivalRate) + 0.5
        elif self.distributionArrival == "Normal":
            pArrRate = abs(np.random.normal(self.pckArrivalRate, 0.6))
            if pArrRate > self.tMax:
                pArrRate = self.tMax
            return pArrRate
        elif self.distributionArrival == "Normal2":
            pArrRate = abs(np.random.normal(self.pckArrivalRate, 0.05))
            if pArrRate > self.tMax:
                pArrRate = self.tMax
            return pArrRate
        elif self.distributionArrival == "Normal3":
            pArrRate = abs(np.random.normal(self.pckArrivalRate, 1))
            if pArrRate > self.tMax:
                pArrRate = self.tMax
            return pArrRate
        elif self.distributionArrival == "Lognormal":
            pArrRate = np.random.lognormal(self.pckArrivalRate, 0.722)
            if pArrRate > self.tMax:
                pArrRate = self.tMax
        elif self.distributionArrival == "Weibull":
            pArrRate = np.random.weibull(self.pckArrivalRate)
            if pArrRate > self.tMax:
                pArrRate = self.tMax
        elif self.distributionArrival == "Beta":
            pArrRate = np.random.beta(self.pckArrivalRate, 1)
            if pArrRate > self.tMax:
                pArrRate = self.tMax
        elif self.distributionArrival == "Gamma":
            pArrRate = np.random.gamma(self.pckArrivalRate, 1)
            if pArrRate > self.tMax:
                pArrRate = self.tMax
        elif self.distributionArrival == "Triangular":
            pArrRate = np.random.triangular(0, self.pckArrivalRate, 1)
            while pArrRate > self.tMax:
                pArrRate = np.random.triangular(0, self.pckArrivalRate, 1)
        elif self.distributionArrival == "Poisson":
            pArrRate = np.random.poisson(self.pckArrivalRate)
            while pArrRate > self.tMax:
                pArrRate = np.random.poisson(self.pckArrivalRate)
        elif self.distributionArrival == "Binomial":
            pArrRate = np.random.binomial(1, self.pckArrivalRate)
            while pArrRate > self.tMax:
                pArrRate = np.random.binomial(1, self.pckArrivalRate)
        elif self.distributionArrival == "Geometric":
            pArrRate = np.random.geometric(self.pckArrivalRate)
            while pArrRate > self.tMax:
                pArrRate = np.random.geometric(self.pckArrivalRate)
        elif self.distributionArrival == "NegativeBinomial":
            pArrRate = np.random.negative_binomial(1, self.pckArrivalRate)
            while pArrRate > self.tMax:
                pArrRate = np.random.negative_binomial(1, self.pckArrivalRate)

        return pArrRate

    def setMeassures(self, tsim):
        """This method calculates average PLR and throughput for the simulation."""
        self.meassuredKPI["PacketLossRate"] = (
            float(100 * self.lostPackets) / self.sentPackets
        )
        if tsim > 1000:
            self.meassuredKPI["Throughput"] = (float(self.rcvdBytes) * 8000) / (
                0.83 * tsim * 1024 * 1024
            )
        else:
            self.meassuredKPI["Throughput"] = 0


class Packet:
    """This class is used to model packets properties and behaviour."""

    def __init__(self, sn, s, qfi, u):
        self.secNum = sn
        self.size = s
        self.qosFlowId = qfi
        self.ue = u
        self.tIn = 0
        self.timestamp = 0
        self.deadline = None
        self.deadline_missed = False
        self.scheduled_at = None
        self.scheduling_delay = None

    def printPacket(self):
        print(
            Format.CYELLOW
            + Format.CBOLD
            + self.ue
            + "+packet "
            + str(self.secNum)
            + " arrives at t ="
            + str(now())
            + Format.CEND
        )


class Bearer:
    """This class is used to model Bearers properties and behaviour."""

    def __init__(self, i, q, tp):
        self.id = i
        self.qci = q
        self.type = tp
        self.buffer = PcktQueue()


class PcktQueue:
    """This class is used to model application and bearer buffers."""

    def __init__(self):
        self.pckts = deque([])

    def insertPckt(self, p):
        self.pckts.append(p)

    def insertPcktLeft(self, p):
        self.pckts.appendleft(p)

    def removePckt(self):
        if len(self.pckts) > 0:
            return self.pckts.popleft()


class RadioLink:
    """This class is used to model radio link properties and behaviour."""

    def __init__(self, i, lq_0, u):
        self.id = i
        state = "ON"
        self.linkQuality = lq_0
        self.ue = u
        self.totCount = 0
        self.maxVar = 0  # 0.1

    def updateLQ(self, env, udIntrv, tSim, fl, u, r):
        """This method updates UE link quality in terms of SINR during the simulation. This is a PEM method.
        During the simulation it is assumed that UE SINR varies following a normal distribution with mean value equal to initial SINR value, and a small variance.
        """
        while env.now < (tSim * 0.83):
            yield env.timeout(udIntrv)
            deltaSINR = random.normalvariate(0, self.maxVar)
            while deltaSINR > self.maxVar or deltaSINR < (0 - self.maxVar):
                deltaSINR = random.normalvariate(0, self.maxVar)
            self.linkQuality = self.linkQuality + deltaSINR


class Format:
    CEND = "\33[0m"
    CBOLD = "\33[1m"
    CITALIC = "\33[3m"
    CURL = "\33[4m"
    CBLINK = "\33[5m"
    CBLINK2 = "\33[6m"
    CSELECTED = "\33[7m"
    CBLACK = "\33[30m"
    CRED = "\33[31m"
    CGREEN = "\33[32m"
    CYELLOW = "\33[33m"
    CBLUE = "\33[34m"
    CVIOLET = "\33[35m"
    CBEIGE = "\33[36m"
    CWHITE = "\33[37m"
    CGREENBG = "\33[42m"
    CBLUEBG = "\33[44m"
