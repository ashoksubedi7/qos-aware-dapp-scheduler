"""This module contains the UE, Packet Flow, Packet, PcktQueue, Bearer and RadioLink classes.
These classes are oriented to describe UE traffic profile, and UE relative concepts.
"""
import hashlib
import os
import random
import sys
from collections import deque
from scipy.stats import truncnorm, lognorm
from scipy.stats import truncexpon
from scipy.stats import uniform
import numpy as np
import simpy
from environment.radio_scenarios import RadioScenario

def stable_traffic_seed(
    experiment_seed,
    slice_name,
    ue_id,
    direction,
    flow_id,
):
    """
    Create a deterministic per-flow traffic seed.

    Python's built-in hash() is intentionally not used
    because its value can vary between interpreter runs.
    """

    payload = (
        "traffic"
        f"|{int(experiment_seed)}"
        f"|{slice_name}"
        f"|{ue_id}"
        f"|{direction}"
        f"|{int(flow_id)}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

# UE class: terminal description
class UE:
    """This class is used to model UE behaviour and relative properties"""

    def __init__(self, i, ue_sinr0, p, npM, experiment_config=None, slice_name=None, t_sim=None, radio_update_interval=None):
        self.id = i
        self.state = "RRC-IDLE"
        self.packetFlows = []
        self.bearers = []
        self.radioLinks = RadioLink(1, ue_sinr0, self.id, experiment_config=experiment_config, slice_name=slice_name, t_sim=t_sim, update_interval=radio_update_interval)
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
            self.packetFlows[0].recordPacketDrop(
                pD.secNum
            )

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
        experiment_config=None,
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
        self.deadlineEvaluatedPacketIds = set()
        self.deadlineMissPacketIds = set()
        self.deliveredPackets = 0
        self.deliveredBytes = 0
        self.activeDeliveredBytes = 0
        self.drainDeliveredBytes = 0
        self.activeMeasurementEndTime = None

        # Packet lifecycle accounting.
        self.generatedPacketIds = set()
        self.droppedPacketIds = set()
        self.completedPacketIds = set()

        self.packetGenerationTimes = {}
        self.completionDelays = []
        self.experiment_config = (
            experiment_config
        )
        if self.experiment_config is None:
            # Preserve historical simulator behavior.
            self.py_rng = random
            self.np_rng = np.random
            self.traffic_seed = None

        else:
            self.traffic_seed = (
                stable_traffic_seed(
                    experiment_seed=(
                        self.experiment_config.seed
                    ),
                    slice_name=self.sliceName,
                    ue_id=self.ue,
                    direction=self.type,
                    flow_id=self.id,
                )
            )

            self.py_rng = random.Random(
                self.traffic_seed
            )

            self.np_rng = (
                np.random.default_rng(
                    self.traffic_seed
                )
            )


    def _recordDeadlineOutcome(
        self,
        packet,
        missed,
    ):
        """Record a scheduling-deadline outcome exactly once.

        Returns True only when a new deadline outcome is recorded for this
        packet.
        """
        if packet.deadline is None:
            return False
        packet_id = int(packet.secNum)
        if packet_id in self.deadlineEvaluatedPacketIds:
            return False
        self.deadlineEvaluatedPacketIds.add(packet_id)
        self.deadlineEvaluated += 1
        packet.deadline_missed = bool(missed)
        if missed:
            self.deadlineMissPacketIds.add(packet_id)
            self.deadlineMisses += 1
        return True

    def recordDeadlineCrossing(
        self,
        packet,
        now,
    ):
        """Record an unscheduled packet as a deadline miss once its scheduling
        deadline has been exceeded.

        A packet exactly at its deadline is not late.
        """
        if packet.deadline is None:
            return False
        if packet.scheduled_at is not None:
            return False
        packet_id = int(packet.secNum)
        if packet_id in self.deadlineEvaluatedPacketIds:
            return False
        generation_time = self.packetGenerationTimes.get(
            packet_id,
            packet.tIn,
        )
        elapsed = float(now) - float(generation_time)
        if elapsed <= float(packet.deadline):
            return False
        return self._recordDeadlineOutcome(
            packet,
            missed=True,
        )

    def recordUnscheduledDropDeadlineFailure(
        self,
        packet_id,
    ):
        """Record a deadline failure when a deadline-bearing packet is permanently
        dropped before receiving its first scheduling decision.

        A permanently dropped unscheduled packet can no longer satisfy its
        scheduling deadline.
        """
        if self.schedulingDeadline is None:
            return False
        packet_id = int(packet_id)
        if packet_id in self.deadlineEvaluatedPacketIds:
            return False
        self.deadlineEvaluatedPacketIds.add(packet_id)
        self.deadlineMissPacketIds.add(packet_id)
        self.deadlineEvaluated += 1
        self.deadlineMisses += 1
        return True


    def recordSchedulingOutcome(
        self,
        packet,
        now,
    ):
        """
        Record first-scheduling delay exactly once.

        Deadline accounting is also exactly once. A packet
        may already have been classified as a miss by the
        deadline-crossing monitor before it is eventually
        scheduled.
        """

        if packet.scheduled_at is not None:
            return

        packet.scheduled_at = float(
            now
        )

        packet.scheduling_delay = (
            float(now)
            - float(packet.tIn)
        )

        packet_id = int(
            packet.secNum
        )

        self.packetGenerationTimes.setdefault(
            packet_id,
            float(packet.tIn),
        )

        if packet.deadline is None:
            return

        self.schedulingDelays.append(
            packet.scheduling_delay
        )

        missed = (
            packet.scheduling_delay
            > float(packet.deadline)
        )

        self._recordDeadlineOutcome(
            packet,
            missed=missed,
        )


    def recordDeliveryCompletion(
        self,
        packet_id,
        completion_time,
    ):
        """
        Record final successful packet delivery.

        Returns the generation-to-completion delay for
        the first valid completion. Duplicate or unknown
        completions return None.
        """
        packet_id = int(packet_id)

        if (
            packet_id
            in self.completedPacketIds
        ):
            return None
        if (
            packet_id
            in self.droppedPacketIds
        ):
            raise ValueError(
                "dropped packet cannot later "
                "be completed"
            )

        generation_time = (
            self.packetGenerationTimes.get(
                packet_id
            )
        )

        if generation_time is None:
            return None

        completion_delay = (
            float(completion_time)
            - float(generation_time)
        )

        self.completedPacketIds.add(
            packet_id
        )

        self.deliveredPackets += 1

        self.completionDelays.append(
            completion_delay
        )

        del self.packetGenerationTimes[
            packet_id
        ]

        return completion_delay


    def recordPacketDrop(
        self,
        packet_id,
    ):
        """
        Record a true packet drop exactly once.

        End-of-simulation residual packets are not
        considered drops.
        """

        packet_id = int(
            packet_id
        )

        if (
            packet_id
            in self.completedPacketIds
        ):
            raise ValueError(
                "cannot drop an already "
                "completed packet"
            )

        if (
            packet_id
            in self.droppedPacketIds
        ):
            return False
        if (
            packet_id
            not in self.deadlineEvaluatedPacketIds
        ):
            self.recordUnscheduledDropDeadlineFailure(
                packet_id
            )

        self.droppedPacketIds.add(
            packet_id
        )

        self.lostPackets += 1

        self.packetGenerationTimes.pop(
            packet_id,
            None,
        )

        return True

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

    def getActiveMeasurementEndTime(
        self,
        tSim,
    ):
        """
        Return the end of the experiment's active
        measurement window.

        For legacy simulations this preserves the
        historical 0.83 boundary.

        For AssuredQoS this is the start of the
        explicit drain interval.
        """

        tSim = float(tSim)

        if tSim <= 0:
            raise ValueError(
                "tSim must be greater "
                "than zero"
            )

        if self.experiment_config is None:
            return (
                tSim
                * 0.83
            )

        drain_duration = float(
            self.experiment_config
            .drain_duration_ms
        )

        if drain_duration >= tSim:
            raise ValueError(
                "drain_duration_ms must be "
                "smaller than tSim"
            )

        return (
            tSim
            - drain_duration
        )


    def getTrafficEndTime(
        self,
        tSim,
    ):
        """
        Return the time at which this flow stops
        generating new application packets.
        """

        tSim = float(tSim)

        if tSim <= 0:
            raise ValueError(
                "tSim must be greater "
                "than zero"
            )

        if self.experiment_config is None:
            return (
                tSim
                * self.deactivationTime
                * 0.83
            )

        active_end = (
            self.getActiveMeasurementEndTime(
                tSim
            )
        )

        flow_end = (
            tSim
            * self.deactivationTime
        )

        return min(
            flow_end,
            active_end,
        )


    def queueAppPckt(self, env, tSim):  # --- PEM -----
        """This method creates packets according to the packet flow traffic profile and stores them in the application buffer."""
        ueN = int(self.ue[2:])  # number of UEs in simulation
        self.activeMeasurementEndTime = (
            self.getActiveMeasurementEndTime(
                tSim
            )
        )

        end_time = (
            self.getTrafficEndTime(
                tSim
            )
        )

        self.tStart = (
            self.py_rng.expovariate(
                1.0
            )
            + tSim
            * self.activationTime
        )

        yield env.timeout(
            self.tStart
        )
        while env.now < end_time:
            self.sentPackets = self.sentPackets + 1
            size = self.getPsize()
            pD = Packet(self.pId, size + self.header, self.qosFlowId, self.ue)
            packet_id = int(
                pD.secNum
            )

            pD.tIn = float(
                env.now
            )

            pD.timestamp = float(
                env.now
            )

            pD.deadline = (
                self.schedulingDeadline
            )

            self.generatedPacketIds.add(
                packet_id
            )

            self.packetGenerationTimes[
                packet_id
            ] = float(
                pD.tIn
            )

            self.pId = self.pId + 1

            self.appBuff.insertPckt(
                pD
            )
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
        normal_samples = self.np_rng.normal(size=size)
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
            pSize = self.py_rng.paretovariate(1.2) * (0.2 / 1.2) * 2 + self.packetSize
            return int(pSize)
        elif self.distributionSize == "Pareto2":
            maximum = 700
            alpha = 1.2
            mean = self.packetSize
            size = 1
            k = (alpha - 1) * mean / maximum

            pareto_samples = maximum * (self.np_rng.pareto(alpha, size=size) + k)
            truncated_samples = np.clip(pareto_samples, None, maximum)

            pSize = truncated_samples
            return int(pSize)
        elif self.distributionSize == "Lognormal":
            std = 1
            X = self.packetSize
            mu = np.log(X**2 / np.sqrt(X**2 + std**2))
            sigma = np.sqrt(np.log(1 + (std**2 / X**2)))
            pSize = self.np_rng.lognormal(mu, sigma)

            if pSize > self.sMax:
                pSize = self.sMax

            return int(pSize)
        elif self.distributionSize == "Constant":
            return self.packetSize
        elif self.distributionSize == "Uniform":
            pSize = self.py_rng.uniform(self.packetSize, self.sMax)
            return int(pSize)
        elif self.distributionSize == "TruncatedNormal":
            pSize = int(self.truncated_lognormal(self.packetSize, 3, 900))
            return pSize
        elif self.distributionSize == "Normal":
            pSize = self.np_rng.normal(self.packetSize, 40)
            if pSize > self.sMax:
                pSize = self.sMax
            return pSize
        elif self.distributionSize == "Uniform2":
            a = 500 - 2
            b = 500 + 2
            if self.experiment_config is None:
                return int(
                    uniform.rvs(
                        loc=a,
                        scale=10,
                        size=1,
                    )[0]
                )
            return int(
                uniform.rvs(
                    loc=a,
                    scale=10,
                    size=1,
                    random_state=self.np_rng,
                )[0]
            )
            return int(uniform.rvs(loc=a, scale=10, size=1)[0])
        elif self.distributionSize == "Normal2":
            pSize = self.np_rng.normal(self.packetSize, self.packetSize / 15)
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
            if self.experiment_config is None:
                return int(
                    truncexpon.rvs(
                        b=b,
                        loc=0,
                        scale=scale,
                        size=1,
                    )[0]
                )
            return int(
                truncexpon.rvs(
                    b=b,
                    loc=0,
                    scale=scale,
                    size=1,
                    random_state=self.np_rng,
                )[0]
            )
            return int(truncexpon.rvs(b=b, loc=0, scale=scale, size=1)[0])
        elif self.distributionSize == "Exponential":
            pSize = self.np_rng.exponential(self.packetSize)
            if pSize > self.sMax:
                pSize = self.sMax
            return pSize
        elif self.distributionSize == "Gamma":
            pSize = self.np_rng.gamma(self.packetSize, 0.722)
            if pSize > self.sMax:
                pSize = self.sMax
            return pSize
        elif self.distributionSize == "Weibull":
            pSize = self.np_rng.weibull(self.packetSize)
            while pSize > self.sMax:
                pSize = self.np_rng.weibull(self.packetSize)
            return pSize
        elif self.distributionSize == "Beta":
            pSize = self.np_rng.beta(self.packetSize, 0.722)
            while pSize > self.sMax:
                pSize = self.np_rng.beta(self.packetSize, 0.722)
            return pSize
        else:
            print("Error: Distribution size not defined.")

    def getParrRate(self):
        if self.distributionArrival == "Constant":
            pArrRate = self.pckArrivalRate
        elif self.distributionArrival == "Pareto":
            pArrRate = self.py_rng.paretovariate(1.2) * (
                self.pckArrivalRate * (0.2 / 1.2)
            )
        elif self.distributionArrival == "Pareto2":
            maximum = 1.4
            alpha = 1.2
            mean = self.pckArrivalRate
            size = 1
            k = (alpha - 1) * mean / maximum

            pareto_samples = maximum * (self.np_rng.pareto(alpha, size=size) + k)
            truncated_samples = np.clip(pareto_samples, None, maximum)
            pArrRate = truncated_samples
            return int(pArrRate)
        elif self.distributionArrival == "Exponential":
            pArrRate = self.np_rng.exponential(self.pckArrivalRate)
            if pArrRate > self.tMax:
                pArrRate = self.tMax
        elif self.distributionArrival == "Uniform":
            pArrRate = self.py_rng.uniform(0, self.pckArrivalRate)
            if pArrRate > self.tMax:
                pArrRate = self.tMax
        elif self.distributionArrival == "Uniform2":
            pArrRate = self.py_rng.uniform(0, self.pckArrivalRate) + 0.5
        elif self.distributionArrival == "Normal":
            pArrRate = abs(self.np_rng.normal(self.pckArrivalRate, 0.6))
            if pArrRate > self.tMax:
                pArrRate = self.tMax
            return pArrRate
        elif self.distributionArrival == "Normal2":
            pArrRate = abs(self.np_rng.normal(self.pckArrivalRate, 0.05))
            if pArrRate > self.tMax:
                pArrRate = self.tMax
            return pArrRate
        elif self.distributionArrival == "Normal3":
            pArrRate = abs(self.np_rng.normal(self.pckArrivalRate, 1))
            if pArrRate > self.tMax:
                pArrRate = self.tMax
            return pArrRate
        elif self.distributionArrival == "Lognormal":
            pArrRate = self.np_rng.lognormal(self.pckArrivalRate, 0.722)
            if pArrRate > self.tMax:
                pArrRate = self.tMax
        elif self.distributionArrival == "Weibull":
            pArrRate = self.np_rng.weibull(self.pckArrivalRate)
            if pArrRate > self.tMax:
                pArrRate = self.tMax
        elif self.distributionArrival == "Beta":
            pArrRate = self.np_rng.beta(self.pckArrivalRate, 1)
            if pArrRate > self.tMax:
                pArrRate = self.tMax
        elif self.distributionArrival == "Gamma":
            pArrRate = self.np_rng.gamma(self.pckArrivalRate, 1)
            if pArrRate > self.tMax:
                pArrRate = self.tMax
        elif self.distributionArrival == "Triangular":
            pArrRate = self.np_rng.triangular(0, self.pckArrivalRate, 1)
            while pArrRate > self.tMax:
                pArrRate = self.np_rng.triangular(0, self.pckArrivalRate, 1)
        elif self.distributionArrival == "Poisson":
            pArrRate = self.np_rng.poisson(self.pckArrivalRate)
            while pArrRate > self.tMax:
                pArrRate = self.np_rng.poisson(self.pckArrivalRate)
        elif self.distributionArrival == "Binomial":
            pArrRate = self.np_rng.binomial(1, self.pckArrivalRate)
            while pArrRate > self.tMax:
                pArrRate = self.np_rng.binomial(1, self.pckArrivalRate)
        elif self.distributionArrival == "Geometric":
            pArrRate = self.np_rng.geometric(self.pckArrivalRate)
            while pArrRate > self.tMax:
                pArrRate = self.np_rng.geometric(self.pckArrivalRate)
        elif self.distributionArrival == "NegativeBinomial":
            pArrRate = self.np_rng.negative_binomial(1, self.pckArrivalRate)
            while pArrRate > self.tMax:
                pArrRate = self.np_rng.negative_binomial(1, self.pckArrivalRate)

        return pArrRate

    def setMeassures(self, tSim):
        """
        Calculate average PLR and throughput
        for the simulation.
        """

        if self.sentPackets > 0:
            self.meassuredKPI[
                "PacketLossRate"
            ] = (
                float(
                    100
                    * self.lostPackets
                )
                / self.sentPackets
            )
        else:
            self.meassuredKPI[
                "PacketLossRate"
            ] = 0.0

        if self.experiment_config is None:
            duration_factor = 0.83
        else:
            duration_factor = 1.0

        if tSim > 1000:
            self.meassuredKPI[
                "Throughput"
            ] = (
                float(
                    self.rcvdBytes
                )
                * 8000
            ) / (
                duration_factor
                * tSim
                * 1024
                * 1024
            )
        else:
            self.meassuredKPI[
                "Throughput"
            ] = 0


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

    def __init__(
        self,
        i,
        lq_0,
        u,
        experiment_config=None,
        slice_name=None,
        t_sim=None,
        update_interval=None,
    ):
        self.id = i
        self.state = "ON"
        self.linkQuality = float(lq_0)
        self.ue = u
        self.totCount = 0
        self.maxVar = 0

        # None means legacy simulator radio behavior.
        self.radio_scenario = None

        if experiment_config is not None:
            if slice_name is None:
                raise ValueError(
                    "slice_name is required when "
                    "experiment_config is supplied"
                )

            if t_sim is None:
                raise ValueError(
                    "t_sim is required when "
                    "experiment_config is supplied"
                )

            if update_interval is None:
                raise ValueError(
                    "update_interval is required when "
                    "experiment_config is supplied"
                )

            self.radio_scenario = RadioScenario(
                scenario=experiment_config.radio_scenario,
                experiment_seed=experiment_config.seed,
                slice_name=slice_name,
                ue_id=u,
                simulation_time_ms=t_sim,
                update_interval_ms=update_interval,
            )

            self.linkQuality = (
                self.radio_scenario.initial_sinr()
            )

    def updateLQ(self, env, udIntrv, tSim, fl, u, r):
        """This method updates UE link quality in terms of SINR during the simulation. This is a PEM method.
        During the simulation it is assumed that UE SINR varies following a normal distribution with mean value equal to initial SINR value, and a small variance.
        """
        if self.radio_scenario is None:
            end_time = tSim * 0.83
        else:
            end_time = tSim

        while env.now < end_time:
            remaining = (
                end_time - env.now
            )

            if remaining <= 0:
                break

            step = min(
                float(udIntrv),
                float(remaining),
            )

            yield env.timeout(
                step
            )

            if self.radio_scenario is not None:
                self.linkQuality = (
                    self.radio_scenario.sinr_at(
                        env.now
                    )
                )
                continue

            deltaSINR = random.normalvariate(
                0,
                self.maxVar,
            )

            while (
                deltaSINR > self.maxVar
                or deltaSINR < -self.maxVar
            ):
                deltaSINR = (
                    random.normalvariate(
                        0,
                        self.maxVar,
                    )
                )

            self.linkQuality = (
                self.linkQuality
                + deltaSINR
            )


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
