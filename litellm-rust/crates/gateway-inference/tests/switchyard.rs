#![cfg(feature = "switchyard")]

mod support;

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use litellm_core::resources::CoreResources;
use litellm_gateway_inference::{Deployment, ModelRouter, switchyard::SwitchyardDecisionRoute};
use litellm_http::{ClientVariant, HttpSettings, Resolution, media::PublicDnsResolver};
use rstest::rstest;
use serde_json::json;
use switchyard_libsy::{Algorithm, Driver, RoutingOutcome, RuntimeModels};
use switchyard_protocol::{Category, LlmResponse, ModelId, Request, completion_text, text_request};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_partial_json, method, path},
};

struct JudgeThenRoute;

#[async_trait]
impl Algorithm for JudgeThenRoute {
    fn name(&self) -> &str {
        "judge_then_route"
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        request: Request,
    ) -> switchyard_libsy::Result<RoutingOutcome> {
        let response = driver
            .call_model(request.clone(), vec![ModelId::from("judge")])
            .await?;
        let LlmResponse::Agg(response) = response.llm_response else {
            panic!("judge response was not buffered");
        };
        assert_eq!(completion_text(&response), "capable");
        Ok(RoutingOutcome::route_to(
            ModelId::from("primary"),
            vec![ModelId::from("fallback")],
            request,
        ))
    }
}

#[rstest]
#[tokio::test]
async fn services_the_judge_and_returns_the_ordered_decision() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "judge-model",
            "messages": [{"role": "user", "content": "fix this"}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "judge-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "capable"}}]
        })))
        .expect(1)
        .mount(&upstream)
        .await;
    let pool = Arc::new(litellm_http::HttpClientPool::new(Arc::new(
        PublicDnsResolver,
    )));
    let resources = CoreResources::new(pool);
    let http = Resolution::from(&HttpSettings::default()).config;
    let client = resources
        .pool
        .client(&http, ClientVariant::Provider)
        .expect("test HTTP client");
    let route = SwitchyardDecisionRoute::new(
        Arc::new(JudgeThenRoute),
        litellm_core::chat_completions::ChatCompletionsRoute::new(
            client,
            resources.auth,
            Arc::new(support::NoSecrets),
        ),
        [(
            "judge".to_string(),
            Deployment {
                model: "judge-model".to_string(),
                api_key: Some("test-key".to_string()),
                api_base: Some(format!("{}/v1", upstream.uri())),
                custom_llm_provider: Some("openai_like".to_string()),
                ..Deployment::default()
            },
        )]
        .into_iter()
        .collect::<ModelRouter>(),
    );
    let models = Arc::new(RuntimeModels::new(HashMap::from([(
        Category::Any,
        vec![ModelId::from("primary"), ModelId::from("fallback")],
    )])));

    let outcome = route
        .decide(
            Request {
                llm_request: text_request(Some("virtual".to_string()), "fix this"),
                ..Request::default()
            },
            models,
        )
        .await
        .expect("routing succeeds");

    assert_eq!(
        outcome.selected_model_ids,
        vec![ModelId::from("primary"), ModelId::from("fallback")]
    );
    assert_eq!(
        outcome.request.llm_request.model.as_deref(),
        Some("primary")
    );
    assert!(outcome.response.is_none());
}
