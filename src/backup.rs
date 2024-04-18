use Plonky2_lib::aggregator::test::{
    batch_ecdsa_cuda_circuit_with_config, construct_ecdsa_data, gen_batch_ecdsa_proof,
};

// fn main() {
//     println!("Hello, world!");
//     batch_ecdsa_circuit_mpi_test(1);
// }

use mpi::request::WaitGuard;
use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

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
        let start_signal: i32 = 1;
        for i in 1..size {
            world.process_at_rank(i).send(&start_signal);
        }

        let proof = gen_batch_ecdsa_proof(
            1,
            &data,
            v_msg_biguint_target,
            v_pk_x_biguint_target,
            v_pk_y_biguint_target,
            v_r_biguint_target,
            v_s_biguint_target,
        );
        data.verify(proof.clone()).unwrap();
    } else {
        let mut start_signal: i32 = 0;
        world.any_process().receive_into(&mut start_signal);
        if start_signal == 1 {
            let proof = gen_batch_ecdsa_proof(
                batch_num,
                &data,
                v_msg_biguint_target,
                v_pk_x_biguint_target,
                v_pk_y_biguint_target,
                v_r_biguint_target,
                v_s_biguint_target,
            );
            data.verify(proof.clone()).unwrap();
        }
    }
}
