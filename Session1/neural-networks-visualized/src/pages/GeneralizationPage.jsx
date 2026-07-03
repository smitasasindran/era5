import ExperimentLayout from "../components/layout/ExperimentLayout";
import Card from "../components/common/Card";

function GeneralizationPage() {
  return (
    <ExperimentLayout
      title="Generalization"
      claim="A model's real test is how well it performs on data it has never seen before."
    >
      <Card title="Overview">...</Card>
      <Card title="Interactive Demo">...</Card>
      <Card title="What to Observe">...</Card>
      <Card title="Why It Matters">...</Card>
      <Card title="Summary">...</Card>
    </ExperimentLayout>
  );
}

export default GeneralizationPage;
