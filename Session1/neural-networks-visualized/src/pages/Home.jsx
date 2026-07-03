import { FiActivity, FiLayers, FiBox, FiTrendingUp, FiArrowRight } from "react-icons/fi";

import Card from "../components/common/Card";
import SectionTitle from "../components/common/SectionTitle";
import PrimaryButton from "../components/common/PrimaryButton";

// The four interactive experiments linked from the home page.
const experiments = [
    {
        to: "/activations",
        icon: FiActivity,
        title: "Activation Functions",
        description:
            "Compare ReLU, Sigmoid, Tanh and more — see how each one shapes a network's decision boundary.",
    },
    {
        to: "/depth",
        icon: FiLayers,
        title: "Network Depth",
        description:
            "Stack layers interactively and watch how depth changes what a network can learn.",
    },
    {
        to: "/embeddings",
        icon: FiBox,
        title: "Embeddings",
        description: "Explore how raw inputs get mapped into learned vector spaces.",
    },
    {
        to: "/generalization",
        icon: FiTrendingUp,
        title: "Generalization",
        description:
            "Visualize the tug-of-war between overfitting and underfitting as models train.",
    },
];

function Home() {
    return (
        <div className="mx-auto max-w-6xl">
            {/* Hero */}
            <section className="flex flex-col items-start gap-6 py-10 sm:py-14">
                <span className="rounded-full border border-accent-500/30 bg-accent-500/10 px-3 py-1 text-xs font-medium text-accent-300">
                    ERA V5 · Session 1
                </span>
                <h1 className="text-3xl font-extrabold tracking-tight text-slate-50 sm:text-5xl">
                    See how neural networks{" "}
                    <span className="bg-gradient-to-r from-accent-400 to-cyan-400 bg-clip-text text-transparent">
                        actually think
                    </span>
                </h1>
                <p className="max-w-2xl text-base text-slate-400 sm:text-lg">
                    Four interactive experiments that turn abstract deep learning concepts into
                    visuals you can play with, layer by layer.
                </p>
                <PrimaryButton to="/activations" icon={FiArrowRight}>
                    Start exploring
                </PrimaryButton>
            </section>

            {/* Experiment cards */}
            <section className="pb-16">
                <SectionTitle
                    eyebrow="Experiments"
                    title="Pick a concept to visualize"
                    description="Each card opens an interactive playground built to build intuition, not just show equations."
                />

                <div className="mt-8 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
                    {experiments.map((experiment) => (
                        <Card
                            key={experiment.to}
                            to={experiment.to}
                            icon={experiment.icon}
                            title={experiment.title}
                            description={experiment.description}
                        >
                            <span className="mt-4 inline-flex items-center gap-1 text-sm font-medium text-accent-400 transition-all group-hover:gap-2">
                                Explore <FiArrowRight className="h-4 w-4" />
                            </span>
                        </Card>
                    ))}
                </div>
            </section>
        </div>
    );
}

export default Home;
