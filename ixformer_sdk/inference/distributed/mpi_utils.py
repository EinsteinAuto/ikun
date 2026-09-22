import os

from mpi4py import MPI


def get_world_size(comm=None):
    if comm is None:
        comm = MPI.COMM_WORLD

    return comm.Get_size()


def get_local_rank(comm=None):
    return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])


def get_rank(comm=None):
    if comm is None:
        comm = MPI.COMM_WORLD

    return comm.Get_rank()
