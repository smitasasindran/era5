import { useNavigate } from "react-router-dom";
import { FiArrowLeft } from "react-icons/fi";

/**
 * Shared page layout for every experiment page: a back button, a header
 * (title + claim), and a spaced area for the page's content sections —
 * typically five <Card> elements passed in as children.
 */
function ExperimentLayout({ title, claim, children }) {
    const navigate = useNavigate();

    return (
        <div className="mx-auto max-w-7xl">
            <button
                type="button"
                onClick={() => navigate(-1)}
                className="mb-6 inline-flex items-center gap-2 text-sm font-medium text-slate-400 transition-colors hover:text-slate-100"
            >
                <FiArrowLeft className="h-4 w-4" />
                Back
            </button>

            <header className="mb-10">
                <h1 className="text-3xl font-extrabold tracking-tight text-slate-50 sm:text-4xl">
                    {title}
                </h1>

                {claim && (
                    <p className="mt-4 max-w-3xl rounded-lg border border-accent-500/30 bg-accent-500/10 px-4 py-3 text-base leading-relaxed text-accent-200">
                        <span className="mr-2 font-semibold text-accent-300">Claim:</span>
                        {claim}
                    </p>
                )}
            </header>

            <div className="space-y-6 pb-16">{children}</div>
        </div>
    );
}

export default ExperimentLayout;
