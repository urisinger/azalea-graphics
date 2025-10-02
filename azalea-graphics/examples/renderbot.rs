use std::{net::SocketAddr, thread};

use azalea::{ClientInformation, prelude::*};
use azalea_graphics::{
    app::{App, Args as RendererArgs, RendererHandle},
    plugin::RendererPlugin,
};
use clap::Parser;
use tokio::runtime::Runtime;

#[derive(clap::Parser)]
struct Args {
    #[command(flatten)]
    renderer: RendererArgs,

    #[arg(long, short, default_value = "127.0.0.1:25565")]
    addr: SocketAddr,
}

async fn run_azalea(render_handle: RendererHandle, server_address: SocketAddr) {
    let account = Account::offline("bot");

    ClientBuilder::new()
        .add_plugins(RendererPlugin {
            handle: render_handle,
        })
        .set_handler(handle)
        .start(account, server_address)
        .await
        .unwrap();
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    let server_address = args.addr;
    println!("Connecting to: {}", server_address);

    let (handle, app) = App::new(args.renderer);
    let azalea_thread = thread::spawn(move || {
        let rt = Runtime::new().unwrap();
        rt.block_on(run_azalea(handle, server_address));
    });

    app.run();

    let _ = azalea_thread.join();
}

#[derive(Component, Default, Clone)]
struct State;

async fn handle(bot: Client, event: azalea::Event, _state: State) -> anyhow::Result<()> {
    match event {
        azalea::Event::Init => {
            bot.set_client_information(ClientInformation {
                view_distance: 32,
                ..Default::default()
            })
            .await
        }
        _ => {}
    }
    Ok(())
}
