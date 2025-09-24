use std::{env, thread};

use azalea::{ClientInformation, prelude::*};
use azalea_graphics::{
    app::{App, RendererHandle},
    plugin::RendererPlugin,
};
use tokio::runtime::Runtime;

async fn run_azalea(render_handle: RendererHandle, server_address: String) {
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

    let args: Vec<String> = env::args().collect();

    let server_address = if args.len() >= 2 {
        if args[1].parse::<u16>().is_ok() {
            format!("localhost:{}", args[1])
        } else {
            args[1].clone()
        }
    } else {
        "localhost:25565".to_string()
    };

    println!("Connecting to: {}", server_address);

    let (handle, app) = App::new();
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
        azalea::Event::Init => bot.set_client_information(ClientInformation {
            view_distance: 32,
            ..Default::default()
        }).await,
        _ => {}
    };
    Ok(())
}
