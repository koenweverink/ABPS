import logging

# Write INFO+ messages to simulation.log, DEBUG goes nowhere by default
logging.basicConfig(
    filename="simulation.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
)
logger = logging.getLogger("Sim")