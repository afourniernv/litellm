use litellm_host::observation::ObservationSender;
pub mod route;
pub mod types;
pub use litellm_inference::RouteError as Error;
mod common_utils;
pub mod constants;
pub(crate) mod handler;
mod prepare;
use litellm_llms_types::formats::chat_completions::ChatCompletionsResponse;
use prepare::{prepare_provider_request, resolve_output_request, resolve_request};

use litellm_auth::AuthServices;
use litellm_secrets::source::SecretSource;
use std::sync::Arc;
use types::{ChatCompletionsOutput, ChatCompletionsRequest};

#[derive(Clone)]
pub struct ChatCompletionsRoute {
    http: litellm_http::Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
    cache: Option<litellm_cache_response::ScopedCache>,
}

impl ChatCompletionsRoute {
    pub fn new(
        http: litellm_http::Client,
        auth: Arc<AuthServices>,
        secrets: Arc<dyn SecretSource>,
    ) -> Self {
        Self {
            http,
            auth,
            secrets,
            cache: None,
        }
    }

    pub fn with_cache(self, cache: litellm_cache_response::ScopedCache) -> Self {
        Self {
            cache: Some(cache),
            ..self
        }
    }

    pub async fn execute(
        &self,
        request: ChatCompletionsRequest<'_>,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        options: impl Into<litellm_inference::CallOptions>,
    ) -> Result<ChatCompletionsResponse, Error> {
        let litellm_inference::CallOptions {
            cache: cache_options,
            observers,
        } = options.into();
        litellm_host::lifecycle::observe_unary(
            observers.clone(),
            self.run_call(
                request.into(),
                cache_options,
                interceptors,
                observers.as_ref(),
            ),
        )
        .await
    }

    /// Executes a buffered request or returns raw OpenAI-compatible SSE bytes.
    /// Streaming is limited to `openai_like` and bypasses the response cache.
    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "chat_completions",
        model = %request.model,
        provider,
        resolved_model,
        stream,
        outcome
    ))]
    pub async fn execute_output(
        &self,
        request: ChatCompletionsRequest<'_>,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        options: impl Into<litellm_inference::CallOptions>,
    ) -> Result<ChatCompletionsOutput, Error> {
        let litellm_inference::CallOptions {
            cache: cache_options,
            observers,
        } = options.into();
        litellm_host::lifecycle::observe_call(
            observers.clone(),
            litellm_inference::diagnostic::call(self.run(
                request,
                cache_options,
                interceptors,
                observers.as_ref(),
                true,
            )),
        )
        .await
    }

    async fn run(
        &self,
        request: ChatCompletionsRequest<'_>,
        cache_options: Option<litellm_cache_response::CachePolicy>,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        observers: Option<&ObservationSender>,
        allow_raw_stream: bool,
    ) -> Result<ChatCompletionsOutput, Error> {
        let resolved = if allow_raw_stream {
            resolve_output_request(request)?
        } else {
            resolve_request(request)?
        };
        let snapshot = self
            .secrets
            .resolve(&resolved.config.secret_names())
            .await?;
        let prepared = prepare_provider_request(resolved, snapshot)?;
        litellm_inference::diagnostic::provider(&prepared.model, &prepared.custom_llm_provider);
        let execute: futures_util::future::BoxFuture<'_, Result<ChatCompletionsOutput, Error>> =
            Box::pin(handler::execute(
                &self.http,
                &self.auth,
                prepared,
                self.cache.as_ref(),
                cache_options,
                interceptors,
                observers,
            ));
        execute.await
    }
}
