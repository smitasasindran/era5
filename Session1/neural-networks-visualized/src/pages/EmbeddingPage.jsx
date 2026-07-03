import ExperimentLayout from "../components/layout/ExperimentLayout";
import Card from "../components/common/Card";

function EmbeddingPage() {
  return (
    <ExperimentLayout
      title="Embeddings"
      claim="Learned embeddings place similar inputs closer together in vector space."
    >
      <Card title="Overview">...</Card>
      <Card title="Interactive Demo">...</Card>
      <Card title="What to Observe">...</Card>
      <Card title="Why It Matters">...</Card>
      <Card title="Summary">...</Card>
    </ExperimentLayout>
  );
}

export default EmbeddingPage;
