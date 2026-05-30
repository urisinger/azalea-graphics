//! Resolve a Minecraft server address into an IP address and port.

use std::{
    net::{IpAddr, SocketAddr},
    sync::LazyLock,
};

pub use hickory_resolver::net::NetError as ResolveError;
use hickory_resolver::{
    Resolver, TokioResolver,
    config::{GOOGLE, ResolverConfig},
    net::runtime::TokioRuntimeProvider,
    proto::rr::{Name, RData},
};
use tracing::warn;

use crate::address::ServerAddr;

#[doc(hidden)]
#[deprecated(note = "Renamed to ResolveError")]
pub type ResolverError = ResolveError;

static RESOLVER: LazyLock<TokioResolver> = LazyLock::new(|| {
    Resolver::builder_tokio()
        .unwrap_or_else(|_| {
            warn!("System DNS resolver unavailable; falling back to Google DNS.");

            Resolver::builder_with_config(
                ResolverConfig::udp_and_tcp(&GOOGLE),
                TokioRuntimeProvider::new(),
            )
        })
        .build()
        .unwrap()
});

/// Resolve a Minecraft server address into an IP address and port.
///
/// If it's already an IP address, it's returned as-is.
pub async fn resolve_address(mut address: &ServerAddr) -> Result<SocketAddr, ResolveError> {
    let redirect = resolve_srv_redirect(address).await;
    if let Ok(redirect_target) = &redirect {
        address = redirect_target;
    }

    resolve_ip_without_redirects(address).await
}

async fn resolve_ip_without_redirects(address: &ServerAddr) -> Result<SocketAddr, ResolveError> {
    if let Ok(ip) = address.host.parse::<IpAddr>() {
        // no need to do a lookup
        return Ok(SocketAddr::new(ip, address.port));
    }

    let name = Name::from_ascii(&address.host)?;
    let lookup_ip = RESOLVER.lookup_ip(name).await?;

    let ip = lookup_ip
        .iter()
        .next()
        .ok_or(ResolveError::from("No A/AAAA record found"))?;

    Ok(SocketAddr::new(ip, address.port))
}

async fn resolve_srv_redirect(address: &ServerAddr) -> Result<ServerAddr, ResolveError> {
    if address.port != 25565 {
        return Err(ResolveError::from("Port must be 25565 to do a SRV lookup"));
    }

    let query = format!("_minecraft._tcp.{}", address.host);
    let res = RESOLVER.srv_lookup(query).await?;

    let srv = res
        .answers()
        .first()
        .ok_or(ResolveError::from("No SRV record found"))?;
    let RData::SRV(srv) = &srv.data else {
        return Err(ResolveError::from(
            "Record returned from SRV lookup wasn't SRV",
        ));
    };

    Ok(ServerAddr {
        host: srv.target.to_ascii(),
        port: srv.port,
    })
}
