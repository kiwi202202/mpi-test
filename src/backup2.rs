extern crate mpi;

use mpi::traits::*;
use Plonky2_lib::aggregator::test::{
    construct_ecdsa_data, deserialize_proof, gen_batch_ecdsa_proof,
};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    let tasks = vec![1];

    let batch_num = 1;
    let (
        data,
        v_msg_biguint_target,
        v_pk_x_biguint_target,
        v_pk_y_biguint_target,
        v_r_biguint_target,
        v_s_biguint_target,
    ) = construct_ecdsa_data(batch_num);

    if rank == 0 {
        for task in tasks {
            for target_rank in 1..world.size() {
                world.process_at_rank(target_rank).send(&task);
            }

            for _ in 1..world.size() {
                let (proof_bytes, status) = world.any_process().receive_vec::<u8>();

                let proof = deserialize_proof(proof_bytes, &data.common);

                println!(
                    "Received proof from rank {}, proof PIS: {:?}",
                    status.source_rank(),
                    proof.public_inputs
                );

                data.verify(proof.clone()).unwrap();
                println!(
                    "Verifed proof from rank {} at rank {}, proof PIS: {:?}",
                    status.source_rank(),
                    rank,
                    proof.public_inputs
                );
            }
        }
    } else {
        let task = world.process_at_rank(0).receive::<i32>();
        let proof = gen_batch_ecdsa_proof(
            batch_num,
            &data,
            v_msg_biguint_target,
            v_pk_x_biguint_target,
            v_pk_y_biguint_target,
            v_r_biguint_target,
            v_s_biguint_target,
        );
        println!(
            "generate proof at rank {}, proof PIS: {:?}",
            rank, proof.public_inputs
        );
        let proof_bytes = proof.to_bytes();
        world.process_at_rank(0).send(&proof_bytes);
    }
}
