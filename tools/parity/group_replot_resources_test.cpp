#include "host/GroupFarmReplotResources.hpp"
#include <iostream>
using namespace group_grow;
using namespace group_grow::farm;
namespace fs = std::filesystem;
template<class Function> void rejects(Function action, char const* message)
{ bool caught = false; try { action(); } catch (std::exception const&) { caught = true; } require(caught, message); }
Demand service(std::string resource, uint64_t work = 100)
{ Demand d; d.resource = std::move(resource); d.required_units = 1; d.work_us = work; d.period_us = 10000000; d.deadline_us = 1000000; d.bounded = true; return d; }
Demand memory(uint64_t amount)
{ Demand d; d.resource = "ram"; d.transient_units = amount; d.bounded = true; return d; }
int main() try
{
    auto const host = profile_detail::host_name(); auto const cpus = profile_detail::logical_cpus();
    require(!host.empty() && cpus, "fixture cannot identify actual execution host/affinity");
    auto const directory = fs::temp_directory_path() / ("xchplot2-replot-resources-" + std::to_string(Clock::now().time_since_epoch().count()));
    fs::create_directory(directory);
    struct Remove { fs::path path; ~Remove() { std::error_code error; fs::remove_all(path, error); } } remove{directory};
    auto const device = storage_device(directory); auto const disk = replot_storage_id(device);
    Profile profile; profile.source = "declared-test-envelopes"; profile.producer = "fixture";
    Resource cpu; cpu.id = "cpu"; cpu.host = host; cpu.kind = ResourceKind::cpu; cpu.capacity = cpus; cpu.capacity_known = true; cpu.conditional_shared = true;
    Resource ram; ram.id = "ram"; ram.host = host; ram.kind = ResourceKind::host_memory;
    ram.capacity = profile_detail::available_memory(); ram.capacity_known = true;
    Resource storage; storage.id = disk; storage.host = host; storage.kind = ResourceKind::storage; storage.capacity = 1;
    storage.capacity_known = storage.conditional_shared = true;
    if (device.drives.empty()) storage.aliases = {"storage:unknown:" + host};
    else for (auto const& drive : device.drives) storage.aliases.push_back("drive:" + drive);
    Resource controller; controller.id = "controller"; controller.host = host; controller.kind = ResourceKind::controller;
    controller.capacity = 1; controller.capacity_known = controller.conditional_shared = true;
    profile.resources = {cpu, ram, storage, controller};
    constexpr uint64_t mib = 1024 * 1024;
    require(ram.capacity > 64 * mib, "fixture cannot admit its bounded local-memory envelope");
    Workload old; old.id = "legacy"; old.format = "chiapos-v1"; old.decoder = "declared-decoder"; old.decoder_version = "fixture";
    old.backend = "cpu"; old.pool_difficulty = "declared"; old.source = "declared-test-only-not-hardware-measurement";
    old.cost = CostKind::serialized; old.bounded = old.scope_complete = true; old.plot_count = 1;
    old.quality_burst = old.future_quality_burst = 1; old.proof_burst = old.future_proof_burst = 2;
    old.deadline_us = 1000000; old.period_us = 10000000;
    for (auto const& id : {std::string("cpu"), disk, std::string("controller")})
    { Use u; u.resource = id; u.required_units = 1; u.quality_us = 100; u.proof_us = 100; u.blocking_us = 100; old.uses.push_back(u); }
    Use held; held.resource = "ram"; held.resident_units = mib; held.transient_units = mib; old.uses.push_back(held); profile.workloads = {old};
    auto& native = profile.native; native.bounded = native.scope_complete = native.covers_future_filter = native.phase_totals_include_controller = true;
    native.host = host; native.k = 18; native.strength = 2; native.reader = "stateless"; native.backend = "cpu"; native.engine = "reference";
    native.max_ram_bytes = 16 * mib; native.max_cpu_threads = cpus; native.max_batch_members = 1; native.max_group_members = 1;
    native.max_construction_us = native.max_verification_us = 100000; native.output_resources = {disk};
#if defined(__linux__) || defined(_WIN32)
    native.verification_cpu_threads = 1;
#endif
    native.construction = {service("cpu"), memory(18 * mib), service(disk), service("controller")};
    native.verification = native.construction; native.supervision = {service("cpu"), memory(10 * mib), service(disk), service("controller")};
    native.construction[0].required_units = native.supervision[0].required_units = std::min<uint64_t>(cpus, 2);
    native.verification[0].required_units = std::min<uint64_t>(cpus, (native.verification_cpu_threads ? native.verification_cpu_threads : cpus) + 1);
    NativeCohort cohort; cohort.max_members = 2;
    cohort.new_farming = {service("cpu"), memory(mib), service(disk), service("controller")};
    cohort.proof_reserve = cohort.new_farming; native.cohorts = {cohort};
    FarmingState state; state.fixed_extent = true; state.config.k = 18; state.config.strength = 2;
    FarmingDirectory destination; destination.directory = file_io::path_text(directory); destination.filesystem = device.filesystem;
    destination.directory_identity = farming_directory_identity(directory); destination.layout.files = 2; destination.layout.total_members = 2;
    destination.layout.group_members = 1; destination.layout.fixed_members = true; destination.storage.capacity_bytes = 1024 * mib;
    state.directories = {destination};
    pos2gpu::GroupOptions args; args.backend = "cpu"; args.construction = "reference"; args.workers = 1; args.max_ram = 16 * mib; args.batch = 65535; args.compute_mode = "off";
    ReplotSupervision supervision{mib, 8 * mib, 1, 100000}; supervision.source_roots = {directory}; native.source_resources = {disk};
    auto request = replot_native_request(args, profile, state, {}, 0, 16 * mib, supervision);
    {
        ReplotVerification sampled{ReplotVerificationMode::sampled, 1, 3};
        require(replot_requires_proof(sampled, 0) && !replot_requires_proof(sampled, 1) && replot_requires_proof(sampled, 2) &&
            !replot_requires_proof(sampled, 3) && replot_requires_proof(sampled, 5), "saved proof schedule changed its global group ordinals");
        require(replot_requires_proof(replot_verification_policy({}), 0) &&
            !replot_requires_proof(ReplotVerification{}, 0), "legacy absence and explicit structural policy were conflated");
        rejects([&] { validate_replot_verification({ReplotVerificationMode::sampled}); }, "empty proof sample schedule accepted");
        rejects([&] { validate_replot_verification({ReplotVerificationMode::strict, 1}); }, "strict mode ignored a sample schedule");
        require(replot_verification_retry_delay(profile.gate, 0) == profile.gate.recovery_hold_ms &&
            replot_verification_retry_delay(profile.gate, profile.gate.max_age_ms) == profile.gate.max_age_ms,
            "verification retries lost their reviewed cooldown or maximum");
        auto exhausted = profile.gate; exhausted.recovery_hold_ms = 0; exhausted.max_age_ms = 1;
        require(replot_verification_retry_delay(exhausted, 0) == 1 && replot_verification_retry_delay(exhausted, UINT64_MAX) == 1,
            "verification retry cooldown overflowed or spun with a zero recovery hold");
    }
    {
        ReplotReview review; review.farm = state; review.inventory_json = review.resources_json = "resource-fixture";
        review.farm.fingerprint = review.farm.seed = std::string(64, '0'); review.farm.farmer = std::string(96, '0'); review.farm.pool = std::string(64, '0');
        review.farm.job_id = std::string(64, 'a'); review.farm.directories[0].layout.families = 1;
        review.farm.estimate.feasible = true; review.farm.directories[0].layout.files = review.farm.directories[0].layout.total_members = 1;
        review.memory_limit_bytes = 64 * mib; review.sources = {0}; review.space.resize(1);
        ReplotBatch batch; batch.sources = 1; batch.final_bound_bytes = 1; review.batches = {batch};
        {
            auto const strict = replot_review_text(review);
            require(strict.starts_with("ordinary-group-replot-review-v1\n") && replot_review_text(parse_replot_review(strict, 64 * mib)) == strict,
                "legacy review bytes changed without prospective root ownership");
            inventory_detail::RootHandle pin(directory); auto const native = inventory_detail::stat(pin.handle);
            InventoryRoot owner{file_io::path_text(pin.path), farming_directory_identity(pin.path), storage_device(pin.path), uint64_t(native.st_dev), uint64_t(native.st_ino)};
            if (!file_io::capture_durable_identity(pin.handle.fd, &owner.durable))
            {
                auto prospective = review; prospective.owned_roots.push_back(owner); prospective.scratch_directories.push_back(owner.path);
                auto const text = replot_review_text(prospective);
                require(text.starts_with("ordinary-group-replot-review-v2\n") && replot_review_text(parse_replot_review(text, 64 * mib)) == text,
                    "migration root owner lost exact explicit scope or native evidence");
                auto invalid = prospective; invalid.owned_roots.push_back(owner);
                rejects([&] { (void)replot_review_text(invalid); }, "extra duplicate native root gained authority");
                invalid = prospective; invalid.owned_roots[0].storage.filesystem += "-changed";
                rejects([&] { (void)replot_review_text(invalid); }, "native root changed its stored destination storage association");
                invalid = prospective; invalid.scratch_directories.push_back(owner.path);
                rejects([&] { (void)replot_review_text(invalid); }, "duplicate reviewed scratch root gained authority");
                invalid = prospective; invalid.owned_roots[0].durable = {};
                rejects([&] { (void)replot_review_text(invalid); }, "new root schema emitted without prospective native evidence");
            }
        }
        ReplotJournal journal; journal.review_digest = std::string(64, 'a'); journal.farm = review.farm;
        journal.phase = ReplotPhase::replacing; journal.retirements = {{ReplotRetirement::active, false}};
        FarmInventory inventory; InventoryPlot source; source.workload_id = "legacy"; source.header_validated = true;
        source.identity.allocated_bytes = 64; source.identity.links = 1; inventory.plots = {source};
        {
            auto structural = review; structural.verification = ReplotVerification{};
            auto const text = replot_review_text(structural);
            require(text.starts_with("ordinary-group-replot-review-v3\n") && replot_review_text(parse_replot_review(text, 64 * mib)) == text,
                "explicit structural policy lost its canonical review representation");
            auto const legacy_text = replot_review_text(review);
            ReplotRevalidation successor{legacy_text, review.inventory_json, review.resources_json, {}, std::string(64, 'a')};
            successor.verification = ReplotVerification{ReplotVerificationMode::sampled, 1, 3};
            auto const successor_text = replot_revalidation_text(successor);
            auto const parsed_successor = parse_replot_revalidation(successor_text, 64 * mib);
            require(successor_text.starts_with("ordinary-group-replot-revalidation-v3\n") &&
                parsed_successor.origin_review == legacy_text && parsed_successor.verification == successor.verification &&
                replot_revalidation_text(parsed_successor) == successor_text && replot_review_text(review) == legacy_text,
                "explicit successor changed its embedded origin or saved proof schedule");
            auto without_proof = profile; without_proof.native.verification.clear(); without_proof.native.max_verification_us = 0;
            without_proof.native.verification_cpu_threads = 0;
            auto const bounded = replot_native_request(args, without_proof, state, {}, 0, 16 * mib, supervision, false);
            require(bounded.verification_cpu_threads == 1 &&
                assess_replot_remaining(without_proof, bounded, inventory, structural, journal, {{"legacy", 1}}, 0).feasible,
                "routine structural policy required a migration proof service or inherited all CPU lanes");
            rejects([&] { (void)replot_stage(without_proof, bounded, {{"legacy", 1}}, 0, false, true, false); },
                "selected proof work silently downgraded without its reviewed envelope");
            structural.verification = ReplotVerification{ReplotVerificationMode::sampled, 0, 2};
            require(assess_replot_remaining(without_proof, bounded, inventory, structural, journal, {{"legacy", 1}}, 0).feasible,
                "unsampled group was charged for another group's proof");
            structural.verification = ReplotVerification{ReplotVerificationMode::sampled, 1, 0};
            rejects([&] { (void)assess_replot_remaining(without_proof, bounded, inventory, structural, journal, {{"legacy", 1}}, 0); },
                "selected initial sample omitted its required proof work");
            structural.verification = ReplotVerification{};
            auto sealed = journal; stream_state::Stamp stamp; stamp.bytes = 1; sealed.verified_stage = stamp;
            rejects([&] { validate_replot_progress(structural, sealed); }, "new policy stage accepted absent legacy-like proof evidence");
            sealed.full_proof_validated = false;
            auto const body = replot_journal_text(sealed);
            require(body.starts_with("ordinary-group-replot-journal-v4\n") && replot_journal_text(parse_replot_journal(body, 64 * mib)) == body,
                "structural completion lost its explicit false evidence");
            auto changed = sealed; ++changed.revision; changed.parent_digest = stream_state::digest(body); changed.full_proof_validated = true;
            require(!valid_replot_revision(sealed, changed), "completed structural work was relabeled as a validated full proof");
            changed.full_proof_validated.reset();
            require(!valid_replot_revision(sealed, changed), "completed structural work dropped evidence into legacy proof semantics");
            ReplotReceipt receipt; receipt.review_digest = sealed.review_digest; receipt.candidates = 1; receipt.replacement = stamp;
            rejects([&] { validate_replot_receipt(structural, sealed.review_digest, 0, {}, receipt); },
                "new policy receipt accepted missing explicit verification evidence");
            receipt.full_proof_validated = false;
            auto const receipt_body = replot_receipt_text(receipt);
            require(receipt_body.starts_with("ordinary-group-replot-receipt-v3\n") &&
                parse_replot_receipt(receipt_body).full_proof_validated == false && replot_receipt_text(parse_replot_receipt(receipt_body)) == receipt_body,
                "structural receipt claimed a positive full proof or lost canonical bytes");
            require(assess_replot_remaining(without_proof, bounded, inventory, structural, sealed, {{"legacy", 1}}, 0,
                ReplotVerification{ReplotVerificationMode::sampled, 1, 0}).feasible,
                "already completed sample was charged again in a stopped suffix");
        }
        auto completed = profile; completed.native.construction[0].work_us = UINT64_MAX / 4;
        stream_state::Stamp sealed; sealed.bytes = 1; journal.verified_stage = sealed;
        {
            auto const legacy = replot_journal_text(journal);
            require(legacy.starts_with("ordinary-group-replot-journal-v1\n") &&
                replot_journal_text(parse_replot_journal(legacy, 64 * mib)) == legacy &&
                !parse_replot_journal(legacy, 64 * mib).full_proof_validated.has_value(), "legacy migration journal bytes changed without native evidence");
            auto native = journal; native.verified_stage->inode = 1;
            native.verified_stage_identity = {file_io::DurableIdentity::Scheme::windows_file_id,
                "ntfs:12345678-1234-1234-1234-123456789abc:1", std::string(32, '1')};
            require(!replot_output_matches(*native.verified_stage, native.verified_stage_identity, *native.verified_stage, {}),
                "recorded native generation accepted missing fresh evidence with unchanged metadata");
            require(replot_output_matches(*native.verified_stage, {}, *native.verified_stage, native.verified_stage_identity),
                "fresh generation evidence changed strict legacy comparison for an unchanged object");
            auto const body = replot_journal_text(native);
            require(body.starts_with("ordinary-group-replot-journal-v3\n") && replot_journal_text(parse_replot_journal(body, 64 * mib)) == body,
                "migration native journal schema lost exact owner evidence");
            auto altered = native; ++altered.revision; altered.parent_digest = stream_state::digest(body);
            altered.verified_stage_identity.object.back() = '2';
            require(!valid_replot_revision(native, altered), "existing stage acquired different native authority in a journal transition");
            altered = native; ++altered.revision; altered.parent_digest = stream_state::digest(body); altered.verified_stage_identity = {};
            require(!valid_replot_revision(native, altered), "existing stage dropped native authority in a journal transition");
            altered = native; ++altered.revision; altered.parent_digest = stream_state::digest(body); altered.published = altered.verified_stage;
            require(!valid_replot_revision(native, altered), "new publication omitted previously captured stage authority");
            altered.published_identity = altered.verified_stage_identity;
            require(valid_replot_revision(native, altered), "publication failed to propagate exact prospective authority");
            auto remounted = altered; remounted.published->device ^= UINT64_C(0x80000000);
            ++remounted.published->ctime;
            auto const original_stage = native.verified_stage;
            require(valid_replot_revision(native, remounted) && remounted.verified_stage == original_stage &&
                replot_journal_text(parse_replot_journal(replot_journal_text(remounted), 64 * mib)) == replot_journal_text(remounted),
                "native device-only publication recovery changed its canonical stage or rename metadata policy");
            auto legacy_before = native; legacy_before.verified_stage_identity = {};
            auto legacy_after = remounted; legacy_after.verified_stage_identity = legacy_after.published_identity = {};
            legacy_after.parent_digest = stream_state::digest(replot_journal_text(legacy_before));
            require(!valid_replot_revision(legacy_before, legacy_after), "legacy publication retroactively normalized a changed device");
            auto wrong = remounted; wrong.published_identity.object.back() = '2';
            require(!valid_replot_revision(native, wrong), "different native publication authority normalized a changed device");
            wrong = remounted; wrong.published_identity = {};
            require(!valid_replot_revision(native, wrong), "missing publication authority normalized a changed device");
            wrong = remounted; ++wrong.published->inode;
            require(!valid_replot_revision(native, wrong), "native publication authority normalized a changed inode");
            wrong = remounted; ++wrong.published->mtime_ns;
            require(!valid_replot_revision(native, wrong), "native publication authority normalized changed contents metadata");
            auto remounted_ready = remounted; ++remounted_ready.revision;
            remounted_ready.parent_digest = stream_state::digest(replot_journal_text(remounted)); remounted_ready.ready = remounted_ready.published;
            remounted_ready.ready_identity = remounted_ready.published_identity; remounted_ready.ready->device ^= UINT64_C(0x80000000);
            require(valid_replot_revision(remounted, remounted_ready), "native readiness could not bind the exact publication after a device change");
            auto changed_ready = remounted_ready; ++changed_ready.ready->ctime;
            require(!valid_replot_revision(remounted, changed_ready), "native readiness weakened immutable publication metadata");
            auto readiness = altered; ++readiness.revision; readiness.parent_digest = stream_state::digest(replot_journal_text(altered)); readiness.ready = readiness.published;
            require(!valid_replot_revision(altered, readiness), "new readiness omitted previously captured publication authority");
            readiness.ready_identity = readiness.published_identity;
            require(valid_replot_revision(altered, readiness), "readiness failed to propagate exact prospective authority");
            native.published = native.ready = native.verified_stage;
            native.published_identity = native.ready_identity = native.verified_stage_identity;
            native.retirements[0].state = ReplotRetirement::retired;
            ReplotReceipt receipt; receipt.review_digest = native.review_digest; receipt.candidates = 1;
            receipt.replacement = *native.ready; receipt.replacement_identity = native.ready_identity;
            auto const receipt_body = replot_receipt_text(receipt);
            require(receipt_body.starts_with("ordinary-group-replot-receipt-v2\n") &&
                replot_receipt_text(parse_replot_receipt(receipt_body)) == receipt_body, "migration receipt lost prospective native authority");
            auto mismatched = receipt; mismatched.replacement_identity.object.back() = '2'; auto untouched = native;
            rejects([&] { complete_replot_batch(review, mismatched, stream_state::digest(replot_receipt_text(mismatched)), untouched); },
                "completed receipt changed its exact ready object evidence");
            complete_replot_batch(review, receipt, stream_state::digest(receipt_body), native);
            require(!replot_journal_native_evidence(native) && !native.verified_stage && !native.published && !native.ready,
                "completed batch retained active native ownership");
            receipt.replacement_identity = {};
            auto const strict = replot_receipt_text(receipt);
            require(strict.starts_with("ordinary-group-replot-receipt-v1\n") && replot_receipt_text(parse_replot_receipt(strict)) == strict,
                "legacy receipt bytes changed without native evidence");
            require(!parse_replot_receipt(strict).full_proof_validated.has_value() &&
                parse_replot_receipt(strict).full_proof_validated.value_or(true), "legacy receipt absence lost its required-proof interpretation");
        }
        require(assess_replot_remaining(completed, request, inventory, review, journal, {{"legacy", 1}}, 0).feasible,
            "successor replayed construction already covered by an exact verified stage");
        journal.verified_stage.reset();
        require(!assess_replot_remaining(completed, request, inventory, review, journal, {{"legacy", 1}}, 0).feasible,
            "unsealed remaining construction escaped its declared service envelope");
        auto conditional = profile; auto& workload = conditional.workloads[0]; workload.cost = CostKind::serialized; workload.response_tail_us = 0;
        workload.quality_burst = workload.proof_burst = workload.future_quality_burst = workload.future_proof_burst = 1;
        workload.period_us = workload.deadline_us = 1000;
        std::erase_if(workload.uses, [&](auto const& use) { return use.resource != "cpu" && use.resource != "ram"; });
        workload.uses[0].quality_us = workload.uses[0].proof_us = 350; workload.uses[0].blocking_us = 0;
        conditional.native.construction[0].work_us = conditional.native.verification[0].work_us = 200;
        conditional.native.supervision[0].work_us = 1;
        require(!assess_replot_remaining(conditional, request, inventory, review, journal, {{"legacy", 1}}, 0).feasible,
            "remaining forecast invented early-retirement capacity without a debt allowance");
        review.reclaim_before_bytes = source.identity.allocated_bytes;
        require(assess_replot_remaining(conditional, request, inventory, review, journal, {{"legacy", 1}}, 0).feasible,
            "fully observable future cohort lost the existing conditional confirmed-unload forecast");
        journal.retirements[0] = {ReplotRetirement::retired, true};
        require(!assess_replot_remaining(conditional, request, inventory, review, journal, {{"legacy", 1}}, 0).feasible,
            "current unconfirmed final registry demand was erased by a future projection");
        require(assess_replot_remaining(conditional, request, inventory, review, journal, {{"legacy", 0}}, 0).feasible,
            "freshly confirmed complete unload did not release per-file arrival demand");
        inventory.plots[0].header_validated = false; journal.retirements[0] = {ReplotRetirement::active, false};
        require(!assess_replot_remaining(conditional, request, inventory, review, journal, {{"legacy", 1}}, 0).feasible,
            "opaque unidentifiable cohort received hypothetical complete-unload credit");
        inventory.plots[0].header_validated = true;
        auto completed_review = review; completed_review.farm.directories[0].layout.files = completed_review.farm.directories[0].layout.total_members = 2;
        ReplotBatch successor_batch; successor_batch.ordinal = 1; successor_batch.first_source = 1; successor_batch.final_bound_bytes = 1;
        completed_review.batches.push_back(successor_batch);
        auto completed_journal = journal; completed_journal.farm = completed_review.farm; completed_journal.batch = 1;
        completed_journal.retired_prefix = 1; completed_journal.receipt_digest = std::string(64, 'b'); completed_journal.retirements.clear();
        completed_journal.verified_stage = sealed;
        require(assess_replot_remaining(completed, request, inventory, completed_review, completed_journal, {{"legacy", 0}}, 1).feasible,
            "completed historical source reductions or construction phases were replayed during suffix assessment");
        completed_journal.published = completed_journal.ready = sealed;
        require(assess_replot_remaining(completed, request, inventory, completed_review, completed_journal, {{"legacy", 0}}, 2).feasible,
            "already published and ready current output replayed construction or proof work");
        auto debt_review = review; debt_review.sources = {0, 1}; debt_review.batches[0].sources = 2;
        auto debt_inventory = inventory; debt_inventory.plots.push_back(source);
        auto debt_profile = conditional; debt_profile.workloads[0].plot_count = 2;
        auto debt_journal = journal; debt_journal.retirements = {{ReplotRetirement::retired, true}, {ReplotRetirement::active, false}};
        require(!assess_replot_remaining(debt_profile, request, debt_inventory, debt_review, debt_journal, {{"legacy", 1}}, 0).feasible,
            "retired predelete debt was forgotten before selecting another active source");
        debt_review.reclaim_before_bytes = 128;
        require(assess_replot_remaining(debt_profile, request, debt_inventory, debt_review, debt_journal, {{"legacy", 1}}, 0).feasible,
            "remaining active source could not use its separately covered outstanding predelete allowance");
        auto growing = conditional; growing.native.cohorts[0].max_members = 1;
        auto larger = growing.native.cohorts[0]; larger.max_members = 2;
        larger.new_farming[0].work_us = larger.proof_reserve[0].work_us = 200; growing.native.cohorts.push_back(larger);
        require(assess_stage(growing, replot_stage(growing, request, {{"legacy", 0}}, 2, false, false, true)).feasible,
            "cohort growth fixture does not fit after independently confirmed source unload");
        auto growing_journal = journal; growing_journal.farm = completed_review.farm;
        require(!assess_replot_remaining(growing, request, inventory, completed_review, growing_journal, {{"legacy", 1}}, 0).feasible,
            "future cohort growth omitted restart supervision with unconfirmed old registries");
    }
    std::map<std::string, std::string> service_hosts{{"harvester", host}, {"farmer", host}, {"full_node", host}, {"solver", host}};
    struct Other { std::string node_id, peer_host; uint64_t files, members; };
    std::vector<Other> outside;
    require_target_service_scope(profile, service_hosts, outside);
    require_target_pool_difficulty(profile, 0);
    auto pooled = profile; pooled.native.min_pool_difficulty = 128; pooled.native.pool_protocol_version = 2;
    require_target_pool_difficulty(pooled, 128, 2);
    require_target_pool_difficulty(pooled, 256, 2);
    rejects([&] { require_target_pool_difficulty(pooled, 64, 2); }, "changed pool difficulty silently reused a full-solver envelope");
    rejects([&] { require_target_pool_difficulty(pooled, 128, 1); }, "changed pool protocol silently reused a full-solver envelope");
    rejects([&] { require_target_pool_difficulty(profile, 128, 2); }, "solo declaration admitted external full-solver arrivals");
    std::ostringstream pooled_json; write_profile_json(pooled_json, pooled);
    auto pooled_roundtrip = parse_profile(profile_detail::JsonReader(pooled_json.str()).read());
    require(pooled_roundtrip.native.min_pool_difficulty == 128 && pooled_roundtrip.native.pool_protocol_version == 2,
        "quantitative pool scope did not survive profile serialization");
    std::ostringstream solo_json; write_profile_json(solo_json, profile);
    require(solo_json.str().find("\"pool_protocol_version\"") == std::string::npos,
        "optional pool scope changed existing solo profile bytes");
    outside.push_back({std::string(64, 'a'), "untrusted-peer-address", 2, 200});
    rejects([&] { require_target_service_scope(profile, service_hosts, outside); }, "untracked receiver work escaped reviewed resources");
    auto external = profile;
    Resource remote_cpu = cpu; remote_cpu.id = "remote-cpu"; remote_cpu.host = "remote-receiver";
    Resource remote_ram = ram; remote_ram.id = "remote-ram"; remote_ram.host = "remote-receiver";
    Resource network = controller; network.id = "network"; network.host = "reviewed-network-path"; network.kind = ResourceKind::network;
    external.resources.insert(external.resources.end(), {remote_cpu, remote_ram, network});
    Workload unscanned = old; unscanned.id = "outside-receiver"; unscanned.plot_count = 2;
    unscanned.external_service_node_id = outside.front().node_id; unscanned.external_service_host = "remote-receiver";
    unscanned.external_service_members = 200;
    auto remote_use = old.uses.front(); remote_use.resource = remote_cpu.id; unscanned.uses.push_back(remote_use);
    auto remote_memory = held; remote_memory.resource = remote_ram.id; unscanned.uses.push_back(remote_memory);
    auto network_use = remote_use; network_use.resource = network.id; unscanned.uses.push_back(network_use);
    external.workloads.push_back(unscanned);
    require_target_service_scope(external, service_hosts, outside);
    validate_inventory_coverage(external, {{"legacy", 1}});
    rejects([&] { validate_inventory_coverage(external, {{"legacy", 1}, {"outside-receiver", 2}}); }, "unscanned receiver became retirement inventory");
    auto reduced = replot_stage(external, request, {{"outside-receiver", 0}}, 0, false, false, false);
    rejects([&] { (void)assess_stage(external, reduced); }, "retirement released unscanned shared service demand");
    auto population = outside; population.front().members = 201;
    rejects([&] { require_target_service_scope(external, service_hosts, population); }, "external member population exceeded reviewed bound");
    population = outside; population.front().files = 3;
    rejects([&] { require_target_service_scope(external, service_hosts, population); }, "external file population exceeded reviewed bound");
    auto omitted = external; omitted.workloads.back().uses.pop_back();
    rejects([&] { require_target_service_scope(omitted, service_hosts, outside); }, "remote receiver omitted shared-service network load");
    auto moved_service = service_hosts; moved_service["solver"] = "remote-receiver";
    rejects([&] { require_target_service_scope(external, moved_service, outside); }, "solver peer moved beyond its reviewed physical resource host");
    external.native.service_hosts["solver"] = "remote-receiver";
    external.native.cohorts.front().proof_reserve = {service(remote_cpu.id), memory(mib), service(network.id)};
    external.native.cohorts.front().proof_reserve[1].resource = remote_ram.id;
    require_target_service_scope(external, moved_service, outside);
    std::ostringstream external_json; write_profile_json(external_json, external);
    auto external_roundtrip = parse_profile(profile_detail::JsonReader(external_json.str()).read());
    require_target_service_scope(external_roundtrip, moved_service, outside);
    omitted = external; omitted.native.cohorts.front().proof_reserve[1].resource = "ram";
    rejects([&] { replot_stage(omitted, request, {}, 0, false, false, false); }, "remote solver CPU reused another host's memory envelope");
    require(request.cpu_threads == 1 && request.batch_members == 1 && request.controller_ram_bytes == mib, "bridge did not bind actual execution caps");
    require(!request.gpu_bytes && !request.standing_gpu_bytes, "CPU migration acquired GPU residency demand");
    require(request.group_members == 1, "bridge did not derive maximum actual reviewed file membership");
    auto larger_group = state; larger_group.directories[0].layout.files = 1; larger_group.directories[0].layout.total_members = 2;
    larger_group.directories[0].layout.group_members = 2;
    rejects([&] { replot_native_request(args, profile, larger_group, {}, 0, 16 * mib, supervision); }, "batch cap one admitted whole group two above reviewed phase bound one");
    auto execute = replot_execution_options(args, request, Clock::now() - std::chrono::milliseconds(1));
    require(execute.batch == 1 && execute.workers == 1 && execute.cancelled() && execute.protected_resources &&
        execute.io_overlap == args.io_overlap && !execute.generation_overlap && !execute.generation_overlap_auto, "effective protected options/deadline not applied");
    require(replot_ready_members(state, 0) == 0 && replot_ready_members(state, 1) == 1 && replot_ready_members(state, 2) == 2, "ready prefix counted unpublished members");
    rejects([&] { replot_ready_members(state, 3); }, "excess ready prefix accepted");
    std::vector<Stage> stages{replot_stage(profile, request, {}, 0, false, false, false, "before"),
        replot_stage(profile, request, {}, 1, true, false, false, "overlap-build"),
        replot_stage(profile, request, {}, 1, false, true, false, "overlap-proof"),
        replot_stage(profile, request, {{"legacy", 0}}, 2, false, false, true, "final-future")};
    require(!stages[0].new_farming.empty() && !stages[0].proof_reserve.empty() && replot_ready_members(state, 0) == 0,
            "empty target skipped live service reserves or fabricated ready membership");
    auto initial_residency = profile;
    initial_residency.native.cohorts[0].new_farming[1].resident_units = mib;
    initial_residency.native.cohorts[0].proof_reserve[1].resident_units = mib;
    auto initial_stage = replot_stage(initial_residency, request, {}, 0, false, false, false, "initial-services");
    require(assess_stage(initial_residency, initial_stage).feasible && initial_stage.new_farming[1].resident_units == mib &&
            initial_stage.proof_reserve[1].resident_units == mib, "bridge omitted standing target services before first replacement");
    initial_residency.resources[1].capacity = 15 * mib;
    require(!assess_stage(initial_residency, initial_stage).feasible, "bridge admitted initial service residency beyond memory capacity");
    auto assessment = assess_transition(profile, stages);
    require(assessment.feasible && assessment.conditional_shared_service, "actual CPU/one-lane disk/controller profile rejected");
    auto exhausted = profile; exhausted.resources[2].conditional_shared = false;
    require(!assess_stage(exhausted, stages[1]).feasible, "partitioned one-lane disk silently shared");
    exhausted = profile; exhausted.workloads[0].cost = CostKind::reservation; exhausted.workloads[0].response_tail_us = 100;
    for (auto& use : exhausted.workloads[0].uses) if (use.required_units)
    { use.reserved_units = 1; use.required_units = use.quality_us = use.proof_us = use.blocking_us = 0; }
    require(!assess_stage(exhausted, stages[1]).feasible, "opaque reservation exhaustion erased");
    exhausted = profile; exhausted.native.construction[2].required_units = 2;
    require(!assess_stage(exhausted, replot_stage(exhausted, request, {}, 1, true, false, false)).feasible, "oversized storage request erased");
    exhausted = profile; exhausted.native.covers_future_filter = false;
    rejects([&] { replot_stage(exhausted, request, {}, 1, true, false, true); }, "native future declaration inferred");
    exhausted = profile; exhausted.native.supervision[1].transient_units = mib;
    rejects([&] { replot_stage(exhausted, request, {}, 0, false, false, false); }, "RPC footprint omitted");
    exhausted = profile; exhausted.native.source_resources.clear();
    rejects([&] { replot_native_request(args, exhausted, state, {}, 0, 16 * mib, supervision); }, "source filesystem mapping omitted");
    exhausted = profile; exhausted.native.supervision.erase(exhausted.native.supervision.begin() + 2);
    rejects([&] { replot_stage(exhausted, request, {}, 0, false, false, false); }, "source I/O omitted from supervision phase");
    auto offered = args; offered.pool_offer_ram = mib;
    rejects([&] { replot_native_request(offered, profile, state, {}, 0, 16 * mib, supervision); }, "uncovered broker offer admitted");
    auto moved = state; moved.directories[0].directory_identity += "changed";
    rejects([&] { replot_native_request(args, profile, moved, {}, 0, 16 * mib, supervision); }, "changed destination identity accepted");
    auto gpu_profile = profile; gpu_profile.native.backend = "gpu"; gpu_profile.native.max_gpu_bytes = 100;
    gpu_profile.native.standing_gpu_bytes = 25;
    Resource gpu; gpu.id = "gpu"; gpu.host = host; gpu.kind = ResourceKind::gpu_compute; gpu.capacity = 1; gpu.capacity_known = true;
    std::string const gpu_id = "host:" + host + "/pci:0000:01:00.0"; gpu.aliases = {gpu_id + "/compute"};
    Resource vram; vram.id = "vram"; vram.host = host; vram.kind = ResourceKind::gpu_memory; vram.capacity = 125; vram.capacity_known = true;
    vram.aliases = {gpu_id + "/memory"}; gpu_profile.resources.push_back(gpu); gpu_profile.resources.push_back(vram);
    gpu_profile.native.gpu_compute_resource = "gpu"; gpu_profile.native.gpu_memory_resource = "vram";
    for (auto* phase : {&gpu_profile.native.construction, &gpu_profile.native.verification, &gpu_profile.native.supervision})
    {
        phase->front().required_units = cpus;
        Demand held; held.resource = "vram"; held.resident_units = 25; held.bounded = true; phase->push_back(held);
    }
    gpu_profile.native.construction.back().transient_units = 100; gpu_profile.native.construction.push_back(service("gpu"));
    auto gpu_args = args; gpu_args.backend = "gpu";
    auto gpu_request = replot_native_request(gpu_args, gpu_profile, state, gpu_id, 100, 16 * mib, supervision);
    require(gpu_request.gpu_bytes == 100 && gpu_request.standing_gpu_bytes == 25, "bridge conflated allocation grant with standing GPU residency");
    std::ostringstream gpu_report; write_replot_request_json(gpu_report, gpu_request);
    require(gpu_report.str().find("\"gpu_bytes\":100,\"standing_gpu_bytes\":25") != std::string::npos, "request report omitted distinct standing GPU residency");
    auto final_gpu = replot_stage(gpu_profile, gpu_request, {{"legacy", 0}}, 2, false, false, true, "gpu-final-future");
    require(assess_stage(gpu_profile, final_gpu).feasible && final_gpu.supervision.back().resident_units == 25,
            "bridge released standing GPU residency at final future stage");
    auto no_gpu_standing = gpu_profile; no_gpu_standing.native.standing_gpu_bytes = 0;
    rejects([&] { replot_native_request(gpu_args, no_gpu_standing, state, gpu_id, 100, 16 * mib, supervision); }, "bridge admitted unknown standing GPU residency");
    std::cout << "migration physical mapping, phase budgets, one-lane coexistence and deadline checks passed\n";
    return 0;
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
