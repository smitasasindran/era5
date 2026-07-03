import ExperimentLayout from "../components/layout/ExperimentLayout";
import Card from "../components/common/Card";

function ActivationPage() {
  return (
    <ExperimentLayout
      title="Activation Functions"
      claim="Non-linear activations let networks learn boundaries a linear model cannot."
    >
      <Card title="Overview">...</Card>
      <Card title="Interactive Demo">...</Card>
      <Card title="What to Observe">...</Card>
      <Card title="Why It Matters">...</Card>
      <Card title="Summary">...</Card>
    </ExperimentLayout>
  );
}

export default ActivationPage;
