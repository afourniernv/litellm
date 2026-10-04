use std::sync::Arc;

use async_trait::async_trait;
use litellm_core::{
    RouteError,
    chat_completions::{ChatCompletionsRoute, types::ChatCompletionsRequest},
};
use litellm_http::transport::Error as TransportError;
use litellm_router::Router as ModelRouter;
use serde_json::Value;
use switchyard_libsy::{Algorithm, RoutingOutcome, RuntimeModels};
use switchyard_llm_client::ClientRouter;
use switchyard_protocol::{
    LlmClientError, LlmResponse, Request, Response, RoutedLlmClient, WireFormat,
};
use switchyard_translation::{decode_aggregated_response, encode_request};

/// Runs a Switchyard decision through LiteLLM's buffered OpenAI Chat path.
///
/// Routing calls do not use the gateway's interceptors, observers, cache, or target prompts.
pub struct SwitchyardDecisionRoute {
    algorithm: Arc<dyn Algorithm>,
    clients: ClientRouter,
}

impl SwitchyardDecisionRoute {
    pub fn new(
        algorithm: Arc<dyn Algorithm>,
        route: ChatCompletionsRoute,
        deployments: ModelRouter,
    ) -> Self {
        Self {
            algorithm,
            clients: ClientRouter::single(Arc::new(LiteLlmClient { route, deployments })),
        }
    }

    pub async fn decide(
        &self,
        request: Request,
        models: Arc<RuntimeModels>,
    ) -> switchyard_libsy::Result<RoutingOutcome> {
        switchyard_llm_client::decide(
            Arc::clone(&self.algorithm),
            self.clients.clone(),
            request,
            models,
        )
        .await
    }
}

struct LiteLlmClient {
    route: ChatCompletionsRoute,
    deployments: ModelRouter,
}

#[async_trait]
impl RoutedLlmClient for LiteLlmClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        let model =
            request
                .llm_request
                .model
                .as_deref()
                .ok_or_else(|| LlmClientError::InvalidRequest {
                    message: "routed request has no model".to_string(),
                })?;
        let deployment =
            self.deployments
                .get(model)
                .ok_or_else(|| LlmClientError::Configuration {
                    message: format!("unknown LiteLLM deployment {model}"),
                })?;
        let Request {
            llm_request,
            metadata,
            ..
        } = request;
        let Value::Object(mut body) = encode_request(&llm_request, WireFormat::OpenAiChat)
            .map_err(|error| LlmClientError::RequestEncoding(error.to_string()))?
        else {
            return Err(LlmClientError::RequestEncoding(
                "OpenAI Chat request is not an object".to_string(),
            ));
        };
        let messages = body.remove("messages").ok_or_else(|| {
            LlmClientError::RequestEncoding("OpenAI Chat request has no messages".to_string())
        })?;
        body.remove("model");
        let response = self
            .route
            .execute(
                ChatCompletionsRequest {
                    model: &deployment.model,
                    messages,
                    optional_params: body,
                    api_key: deployment.api_key.as_deref(),
                    api_base: deployment.api_base.as_deref(),
                    custom_llm_provider: deployment.custom_llm_provider.as_deref(),
                    extra_headers: None,
                    timeout: deployment.timeout,
                },
                &(),
                None,
            )
            .await
            .map_err(map_error)?;
        let body = serde_json::to_value(response)
            .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))?;
        let response = decode_aggregated_response(&body, WireFormat::OpenAiChat)
            .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))?;
        Ok(Response {
            llm_response: LlmResponse::Agg(response),
            metadata,
            upstream_headers: Default::default(),
        })
    }
}

fn map_error(error: RouteError) -> LlmClientError {
    if error.is_request() {
        return LlmClientError::InvalidRequest {
            message: error.to_string(),
        };
    }
    match error {
        RouteError::Transport(TransportError::Http { status, body }) => {
            match axum::http::StatusCode::from_u16(status) {
                Ok(status) => LlmClientError::UpstreamHttp { status, body },
                Err(source) => LlmClientError::InvalidResponse {
                    source: Box::new(source),
                },
            }
        }
        RouteError::Transport(source) => LlmClientError::Transport {
            source: Box::new(source),
        },
        RouteError::InvalidResponse(_) | RouteError::PostCallHook(_) => {
            LlmClientError::InvalidResponse {
                source: Box::new(error),
            }
        }
        RouteError::Auth(_) | RouteError::Http(_) | RouteError::Secret(_) => {
            LlmClientError::Configuration {
                message: error.to_string(),
            }
        }
        RouteError::InvalidType { .. }
        | RouteError::MissingField(_)
        | RouteError::InvalidProvider(_)
        | RouteError::InvalidRequest(_)
        | RouteError::Unsupported(_)
        | RouteError::Headers(_) => LlmClientError::InvalidRequest {
            message: error.to_string(),
        },
    }
}
